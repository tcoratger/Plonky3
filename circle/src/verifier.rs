use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::fs::{DefaultCodec, ExtensionFieldCodec, FieldToFieldCodec, VerifierState};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{BatchDimensions, BatchOpeningRef, Mmcs, MmcsReader};
use p3_field::{ExtensionField, PrimeField};
use p3_fri::protocol::Protocol;
use p3_fri::verifier::FriError;
use p3_fri::{FriFoldingStrategy, FriParameters};
use p3_matrix::Dimensions;

use crate::proof::CircleCommitPhaseProofStep;
use crate::protocol::{CircleLabelsDefault, CircleProtocol, FriLabels};

pub fn verify<'a, Folding, Val, Challenge, M, Challenger, InputError>(
    folding: &Folding,
    params: &FriParameters<M>,
    protocol: &CircleProtocol,
    transcript: &mut VerifierState<'a, Challenger>,
    mut open_input: impl FnMut(
        usize,
        usize,
        &mut VerifierState<'a, Challenger>,
    ) -> Result<Vec<(usize, Challenge)>, FriError<M::Error, InputError>>,
) -> Result<(), FriError<M::Error, InputError>>
where
    Val: PrimeField,
    Challenge: ExtensionField<Val>,
    M: MmcsReader<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger<Witness = Val> + DefaultCodec<M::Digest>,
    Folding: FriFoldingStrategy<Val, Challenge>,
    InputError: core::fmt::Debug,
{
    let fri_mmcs = &params.mmcs;
    transcript.begin_protocol::<()>(CircleLabelsDefault::PROTOCOL);

    // Phase 1: Derive folding challenges
    //
    // In Circle-FRI, the verifier must produce one random challenge (beta)
    // per commit-phase round. Each commitment is observed into the Fiat-Shamir
    // transcript, then a challenge is sampled.
    // This yields exactly as many betas as there are commit-phase rounds.
    let mut commitments = Vec::with_capacity(protocol.rounds.dimensions.len());
    let mut betas = Vec::with_capacity(protocol.rounds.dimensions.len());
    for (round, dims) in protocol.rounds.dimensions.iter().enumerate() {
        // Absorb this round's commitment into the transcript.
        let commitment = fri_mmcs
            .read_commitment(
                transcript,
                CircleLabelsDefault::round_commitment(round),
                &BatchDimensions::single(*dims),
            )?
            .into_inner();
        // Squeeze a field-extension element to use as the folding challenge.
        let beta = transcript
            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                CircleLabelsDefault::beta(round),
            )
            .into_inner();
        commitments.push(commitment);
        betas.push(beta);
    }

    // Absorb the prover's claimed constant polynomial into the transcript.
    // After all folding rounds, the result should reduce to this constant.
    let final_poly =
        transcript
            .next_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                CircleLabelsDefault::final_poly(),
            )?
            .into_inner();

    // Verify proof-of-work: a grinding witness that the prover must compute
    // to raise the cost of brute-forcing query positions.
    transcript.check_pow(
        CircleLabelsDefault::pow_query(params.query_proof_of_work_bits),
        params.query_proof_of_work_bits,
    )?;

    // Phase 2: Query verification.
    //
    // The folding chain starts after first bivariate layer. Query sampling
    // still includes any extra index bits needed by the PCS callback.
    let log_max_height = protocol.log_max_height;
    for query in 0..params.num_queries {
        // Sample a full query index. The PCS callback uses the extra bit to open
        // the original input and first layer; FRI ignores it by shifting below.
        let index = transcript
            .challenge_bits(
                CircleLabelsDefault::query_index(query),
                protocol.query_bits(),
            )
            .into_inner();

        // Open the input polynomials at this query index. The callback returns reduced
        // opening contributions as `(log_height, value)` pairs, sorted by height.
        let ro = open_input(query, index, transcript)?;

        // Read the FRI round openings for this query. For each round, the transcript
        // provides all sibling evaluations except the queried value and the MMCS opening
        // proof for that row.
        let mut commit_phase_openings = Vec::with_capacity(protocol.rounds.log_arities.len());
        for (round, (&log_arity, dimensions)) in protocol
            .rounds
            .log_arities
            .iter()
            .zip(protocol.rounds.dimensions.iter())
            .enumerate()
        {
            let sibling_values = transcript.next_hints::<Challenge, ExtensionFieldCodec<
                Val,
                Challenge,
                FieldToFieldCodec<Val>,
            >>(
                CircleLabelsDefault::sibling_values(query, round),
                (1 << log_arity) - 1,
            )?;
            let opening_proof = fri_mmcs.read_opening_proof(
                transcript,
                CircleLabelsDefault::round_opening_proof(query, round),
                &BatchDimensions::single(*dimensions),
            )?;
            commit_phase_openings.push(CircleCommitPhaseProofStep {
                log_arity: u8::try_from(log_arity).expect("Circle-FRI log arity fits in u8"),
                sibling_values,
                opening_proof,
            });
        }

        // Zip the challenges, commitments, and openings together for folding.
        //
        // Invariant: all three iterators have the same length here.
        // - The challenges are derived from the commitments (one per round).
        // - The openings count was validated to match the commitment count.
        // Plain zip is safe; it cannot silently truncate.
        let fold_data_iter = betas
            .iter()
            .zip(commitments.iter())
            .zip(commit_phase_openings.iter());

        // Walk the FRI folding chain: at each round, verify the Merkle
        // opening against the commitment, then fold the sibling evaluations
        // using the challenge beta to produce the next-round evaluation.
        let folded_eval = verify_query(
            folding,
            params,
            fri_mmcs,
            index >> folding.extra_query_index_bits(),
            fold_data_iter,
            ro,
            log_max_height,
        )?;

        // After all rounds, the polynomial has been folded to a constant.
        // That constant must equal the prover's claimed final polynomial.
        if folded_eval != final_poly {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    transcript.end_protocol::<()>(CircleLabelsDefault::PROTOCOL);
    Ok(())
}

/// One round's worth of data needed to verify a Circle-FRI fold.
///
/// Groups together:
/// - The random folding challenge for this round.
/// - The Merkle commitment to the evaluations on this round's domain.
/// - The prover-supplied sibling values and Merkle opening proof.
type CommitStep<'a, F, M> = (
    (&'a F, &'a <M as Mmcs<F>>::Commitment),
    &'a CircleCommitPhaseProofStep<F, M>,
);

/// Verify one query chain in the Circle-FRI proof.
///
/// Starting from a leaf in the initial evaluation domain, this walks
/// up the folding tree one round at a time:
///
/// ```text
///     domain size:  2^{log_max_height}  →  ...  →  2^{log_blowup}
///     round:              0                           last
/// ```
///
/// At each round:
/// - Roll in any reduced openings whose height matches the current domain.
/// - Reconstruct the full sibling group from the queried evaluation
///   plus the (arity - 1) sibling values provided by the prover.
/// - Verify the Merkle opening against the round commitment.
/// - Fold the sibling group with the challenge beta to produce the
///   parent evaluation for the next round.
///
/// With variable arity, each round may fold by a different factor
/// (2^{log_arity_i} siblings per group).
///
/// # Returns
///
/// The final folded evaluation, which the caller checks against
/// the prover's claimed constant.
fn verify_query<'a, Folding, F, EF, M, InputError>(
    folding: &Folding,
    params: &FriParameters<M>,
    mmcs: &M,
    mut index: usize,
    steps: impl ExactSizeIterator<Item = CommitStep<'a, EF, M>>,
    reduced_openings: Vec<(usize, EF)>,
    log_max_height: usize,
) -> Result<EF, FriError<M::Error, InputError>>
where
    F: PrimeField,
    EF: ExtensionField<F>,
    M: Mmcs<EF> + 'a,
    Folding: FriFoldingStrategy<F, EF>,
    InputError: core::fmt::Debug,
{
    // Running accumulator: starts at zero and accumulates reduced openings
    // and folding results as we walk up the tree.
    let mut folded_eval = EF::ZERO;

    // Reduced openings arrive sorted by height descending.
    // We consume them as the current domain height matches.
    let mut ro_iter = reduced_openings.into_iter().peekable();

    // Current domain size is 2^{log_current_height}; decreases each round.
    let mut log_current_height = log_max_height;

    for (round, ((&beta, comm), opening)) in steps.enumerate() {
        // This round folds 2^{log_arity} siblings into one parent.
        let log_arity = opening.log_arity as usize;
        let arity = 1 << log_arity;

        // Shape check: the prover must supply exactly (arity - 1) siblings.
        // The queried evaluation itself is the remaining one, so the full
        // group has arity elements total.
        //
        //     sibling_values: [s_0, s_1, ..., s_{arity-2}]   (arity - 1 elements)
        //     queried value:  folded_eval                     (1 element)
        //     full group:     arity elements
        if opening.sibling_values.len() != arity - 1 {
            return Err(FriError::SiblingValuesLengthMismatch {
                round,
                expected: arity - 1,
                got: opening.sibling_values.len(),
            });
        }

        // If there are input polynomials evaluated at this domain height,
        // add their contribution before folding. This is the "roll-in" step
        // that combines multiple polynomials into the FRI batch.
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_current_height) {
            folded_eval += ro;
        }

        // Reconstruct the full evaluation group for this node.
        // The queried index within the group tells us where our value sits;
        // the prover's sibling values fill the remaining positions.
        //
        //     arity = 4, index_in_group = 1:
        //     evals = [sibling_0, folded_eval, sibling_1, sibling_2]
        let index_in_group = index % arity;
        let mut evals = vec![EF::ZERO; arity];
        evals[index_in_group] = folded_eval;

        // Fill in siblings at every position except the queried one.
        let mut sibling_idx = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..arity {
            if j != index_in_group {
                evals[j] = opening.sibling_values[sibling_idx];
                sibling_idx += 1;
            }
        }

        // After folding, the domain halves (or shrinks by 2^{log_arity}).
        let log_folded_height = log_current_height - log_arity;

        // Dimensions for the MMCS verification: one matrix of width = arity
        // at the folded height. This tells the Merkle tree the expected shape.
        let dims = &[Dimensions {
            width: arity,
            height: 1 << log_folded_height,
        }];

        // Move from the leaf index to its parent in the folding tree.
        index >>= log_arity;

        // Verify the Merkle opening: the sibling evaluations the prover
        // gave us must be consistent with the round commitment.
        mmcs.verify_batch(
            comm,
            dims,
            index,
            BatchOpeningRef::new(&[evals.clone()], &opening.opening_proof),
        )
        .map_err(FriError::CommitPhaseMmcsError)?;

        // Fold the full sibling group down to a single evaluation using
        // the random challenge beta. This is the core FRI step.
        folded_eval =
            folding.fold_row(index, log_folded_height, log_arity, beta, evals.into_iter());

        // Advance to the next (smaller) domain.
        log_current_height = log_folded_height;
    }

    // After all rounds, we should have folded down to 2^{log_blowup}.
    // If not, the proof has the wrong number of rounds for the domain size.
    if log_current_height != params.log_blowup {
        return Err(FriError::FinalFoldHeightMismatch {
            expected: params.log_blowup,
            got: log_current_height,
        });
    }

    // All input polynomial evaluations should have been consumed during
    // folding. Leftovers mean the proof contains data for heights that
    // were never reached.
    if let Some((next_log_height, _)) = ro_iter.next() {
        return Err(FriError::UnconsumedReducedOpenings {
            next_log_height,
            remaining: 1 + ro_iter.count(),
        });
    }

    Ok(folded_eval)
}
