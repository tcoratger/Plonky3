use alloc::collections::btree_map::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::fs::{
    DefaultCodec, ExtensionFieldCodec, FieldToFieldCodec, TranscriptError, VerifierState,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{BatchDimensions, BatchOpening, BatchOpeningRef, Mmcs, MmcsReader};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, PrimeField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::{log2_strict_usize, reverse_bits_len};
use thiserror::Error;

use crate::protocol::{BatchSpec, FriLabels, FriLabelsDefault, FriProtocol, Protocol};
use crate::{CommitPhaseProofStep, CommitmentWithOpeningPoints, FriFoldingStrategy, FriParameters};

#[derive(Debug, Error)]
pub enum FriError<CommitMmcsErr, InputError>
where
    CommitMmcsErr: core::fmt::Debug,
    InputError: core::fmt::Debug,
{
    #[error("missing initial reduced opening at log height {expected}")]
    MissingInitialReducedOpening { expected: usize },
    #[error("initial reduced opening height mismatch: expected {expected}, got {got}")]
    InitialReducedOpeningHeightMismatch { expected: usize, got: usize },
    #[error("round {round}: sibling values length mismatch: expected {expected}, got {got}")]
    SiblingValuesLengthMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error("final folded height mismatch: expected {expected}, got {got}")]
    FinalFoldHeightMismatch { expected: usize, got: usize },
    #[error(
        "unconsumed reduced openings remain after folding: next log height {next_log_height}, remaining {remaining}"
    )]
    UnconsumedReducedOpenings {
        next_log_height: usize,
        remaining: usize,
    },
    #[error("batch {batch}: opened-values matrix count mismatch: expected {expected}, got {got}")]
    BatchOpenedValuesCountMismatch {
        batch: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "batch {batch}, matrix {matrix}, point {point}: evaluation count mismatch: expected {expected}, got {got}"
    )]
    PointEvaluationCountMismatch {
        batch: usize,
        matrix: usize,
        point: usize,
        expected: usize,
        got: usize,
    },
    #[error("commit phase MMCS error: {0:?}")]
    CommitPhaseMmcsError(CommitMmcsErr),
    #[error("input error: {0:?}")]
    InputError(InputError),
    #[error("final polynomial mismatch: evaluation does not match expected value")]
    FinalPolyMismatch,
    #[error("transcript error: {0}")]
    Transcript(#[from] TranscriptError),
}

/// A chain of FRI input openings allowing a verifier to check a sequence of
/// FRI folds and rolls. The first element of each pair indicates the round of
/// fri in which the input should be rolled in. The second element is the opening.
pub(crate) type FriOpenings<F> = Vec<(usize, F)>;

/// Verify a transcript-backed FRI proof.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `transcript`: The Fiat-Shamir verifier transcript used to read proof data and sample challenges.
/// - `alpha`: The batch combination challenge sampled before entering the FRI protocol.
/// - `commitments_with_opening_points`: A vector of joint commitments to collections of matrices
///   and openings of those matrices at a collection of points. These values are assumed to have
///   already been read from the transcript by the caller.
/// - `input_mmcs`: The input multi-matrix commitment scheme.
#[allow(clippy::too_many_arguments)]
pub fn verify_fri<'a, Folding, Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    transcript: &mut VerifierState<'a, Challenger>,
    alpha: Challenge,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
    input_mmcs: &InputMmcs,
) -> Result<(), FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField + PrimeField,
    Challenge: ExtensionField<Val>,
    InputMmcs: MmcsReader<Val>,
    FriMmcs: MmcsReader<Challenge>,
    Challenger: FieldChallenger<Val>
        + GrindingChallenger<Witness = Val>
        + DefaultCodec<InputMmcs::Digest>
        + DefaultCodec<FriMmcs::Digest>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    // Recover the protocol shape from the already-read input commitments and opening points.
    // The transcript pattern determines the order of bytes, while `Protocol` gives the verifier
    // the dimensions and folding schedule needed to interpret those bytes.
    let batches = commitments_with_opening_points
        .iter()
        .map(|(_, matrices)| BatchSpec::from_matrices(matrices))
        .collect();
    let protocol = FriProtocol::new(params, batches, folding.extra_query_index_bits());

    transcript.begin_protocol::<()>(FriLabelsDefault::PROTOCOL);

    // Generate all of the random challenges for the FRI rounds, checking PoW per round.
    let mut commitments = Vec::with_capacity(protocol.rounds.dimensions.len());
    let mut betas = Vec::with_capacity(protocol.rounds.dimensions.len());
    for (round, dims) in protocol.rounds.dimensions.iter().enumerate() {
        // Read the folding-round commitment, check the PoW witness, then sample the folding
        // challenge. This is the transcript-backed equivalent of observing the commitment,
        // checking the witness, and sampling beta in `verify_fri`.
        let commitment = params
            .mmcs
            .read_commitment(
                transcript,
                FriLabelsDefault::round_commitment(round),
                &BatchDimensions::single(*dims),
            )?
            .into_inner();
        transcript.check_pow(
            FriLabelsDefault::pow_round(params.commit_proof_of_work_bits, round),
            params.commit_proof_of_work_bits,
        )?;
        let beta = transcript
            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(FriLabelsDefault::beta(
                round,
            ))
            .into_inner();
        commitments.push(commitment);
        betas.push(beta);
    }

    // Ensure that the final polynomial has the expected degree and read all coefficients.
    let final_poly = transcript
        .next_extensions::<Val, Challenge, FieldToFieldCodec<Val>>(
            FriLabelsDefault::final_poly(),
            params.final_poly_len(),
        )?
        .into_iter()
        .map(|value| value.into_inner())
        .collect::<Vec<_>>();

    // Check PoW.
    transcript.check_pow(
        FriLabelsDefault::pow_query(params.query_proof_of_work_bits),
        params.query_proof_of_work_bits,
    )?;

    // The log of the final domain size.
    let log_final_height = params.log_blowup + params.log_final_poly_len;
    for query in 0..params.num_queries {
        // For each query, we start by generating the random index.
        let index = transcript
            .challenge_bits(FriLabelsDefault::query_index(query), protocol.query_bits())
            .into_inner();

        // Next we open all polynomials `f` at the relevant index and combine them into our FRI inputs.
        let input_openings = protocol
            .input_batch_dimensions
            .iter()
            .enumerate()
            .map(|(batch, dimensions)| {
                let opened_values = dimensions
                    .iter()
                    .enumerate()
                    .map(|(matrix, dimensions)| {
                        transcript.next_hints::<Val, FieldToFieldCodec<Val>>(
                            FriLabelsDefault::input_opened_values(query, batch, matrix),
                            dimensions.width,
                        )
                    })
                    .collect::<Result<Vec<_>, TranscriptError>>()?;
                let opening_proof = input_mmcs.read_opening_proof(
                    transcript,
                    FriLabelsDefault::input_opening_proof(query, batch),
                    dimensions,
                )?;
                Ok(BatchOpening::new(opened_values, opening_proof))
            })
            .collect::<Result<Vec<_>, TranscriptError>>()?;

        let reduced_openings = open_input::<Val, Challenge, InputMmcs, FriMmcs>(
            params,
            protocol.log_max_height,
            index,
            &input_openings,
            alpha,
            input_mmcs,
            commitments_with_opening_points,
        )?;

        // Read the FRI round openings for this query. For each round, the transcript provides
        // all sibling evaluations except the queried value and the MMCS opening proof for the row.
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
                FriLabelsDefault::sibling_values(query, round),
                (1 << log_arity) - 1,
            )?;
            let opening_proof = params.mmcs.read_opening_proof(
                transcript,
                FriLabelsDefault::round_opening_proof(query, round),
                &BatchDimensions::single(*dimensions),
            )?;
            commit_phase_openings.push(CommitPhaseProofStep {
                log_arity: u8::try_from(log_arity).expect("FRI log arity fits in u8"),
                sibling_values,
                opening_proof,
            });
        }

        // If we queried extra bits, shift them off now.
        let mut domain_index = index >> folding.extra_query_index_bits();

        let fold_data_iter = betas
            .iter()
            .zip(commitments.iter())
            .zip(commit_phase_openings.iter());

        // Starting at the evaluation at `index` of the initial domain,
        // perform FRI folds until the domain size reaches the final domain size.
        // Check after each fold that the pair of sibling evaluations at the current
        // node match the commitment.
        let folded_eval = verify_query(
            folding,
            &params.mmcs,
            &mut domain_index,
            fold_data_iter,
            reduced_openings,
            protocol.log_max_height,
            log_final_height,
        )?;

        // We open the final polynomial at index `domain_index`, which corresponds to evaluating
        // the polynomial at x^k, where x is the 2-adic generator of order `max_height` and k is
        // `reverse_bits_len(domain_index, log_global_max_height)`.
        let x = Val::two_adic_generator(protocol.log_max_height)
            .exp_u64(reverse_bits_len(domain_index, protocol.log_max_height) as u64);

        // Assuming all the checks passed, the final check is to ensure that the folded evaluation
        // matches the evaluation of the final polynomial sent by the prover.

        // Evaluate the final polynomial at x.
        let mut eval = Challenge::ZERO;
        for &coeff in final_poly.iter().rev() {
            eval = eval * x + coeff;
        }

        if eval != folded_eval {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    transcript.end_protocol::<()>(FriLabelsDefault::PROTOCOL);
    Ok(())
}

pub(crate) type CommitStep<'a, F, M> = (
    (
        &'a F, // The challenge point beta used for the next fold of FRI evaluations.
        &'a <M as Mmcs<F>>::Commitment, // A commitment to the FRI evaluations on the current domain.
    ),
    &'a CommitPhaseProofStep<F, M>, // The sibling and opening proof for the current FRI node.
);

/// Verifies a single query chain in the FRI proof. This is the verifier complement
/// to the prover's [`answer_query`] function.
///
/// Given an initial `index` corresponding to a point in the initial domain
/// and a series of `reduced_openings` corresponding to evaluations of
/// polynomials to be added in at specific domain sizes, perform the standard
/// sequence of FRI folds, checking at each step that the group of sibling evaluations
/// matches the commitment.
///
/// With variable arity, each round may fold by a different factor determined by the
/// `log_arity` field in the opening.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `start_index`: The opening index for the unfolded polynomial.
/// - `fold_data_iter`: An iterator containing, for each fold, the beta challenge, polynomial commitment
///   and commitment opening at the appropriate index.
/// - `reduced_openings`: A vector of pairs of a size and an opening. The opening is a linear combination
///   of all input polynomials of that size opened at the appropriate index. Each opening is added into the
///   the FRI folding chain once the domain size reaches the size specified in the pair.
/// - `log_global_max_height`: The log of the maximum domain size.
/// - `log_final_height`: The log of the final domain size.
#[inline]
pub(crate) fn verify_query<'a, Folding, F, EF, M, InputError>(
    folding: &Folding,
    mmcs: &M,
    start_index: &mut usize,
    fold_data_iter: impl ExactSizeIterator<Item = CommitStep<'a, EF, M>>,
    reduced_openings: FriOpenings<EF>,
    log_global_max_height: usize,
    log_final_height: usize,
) -> Result<EF, FriError<M::Error, InputError>>
where
    F: Field,
    EF: ExtensionField<F>,
    M: Mmcs<EF> + 'a,
    Folding: FriFoldingStrategy<F, EF>,
    InputError: core::fmt::Debug,
{
    let mut ro_iter = reduced_openings.into_iter().peekable();

    // These checks are not essential to security,
    // but they should be satisfied by any non malicious prover.
    // ro_iter being empty means that we have committed to no polynomials at all and
    // we need to roll in a polynomial initially otherwise we are just folding a zero polynomial.
    let Some((first_log_height, _)) = ro_iter.peek() else {
        return Err(FriError::MissingInitialReducedOpening {
            expected: log_global_max_height,
        });
    };
    if *first_log_height != log_global_max_height {
        return Err(FriError::InitialReducedOpeningHeightMismatch {
            expected: log_global_max_height,
            got: *first_log_height,
        });
    }
    let mut folded_eval = ro_iter.next().unwrap().1;

    // Track the current log_height as we fold down
    let mut log_current_height = log_global_max_height;

    // We start with evaluations over a domain of size (1 << log_global_max_height). We fold
    // using FRI until the domain size reaches (1 << log_final_height).
    for (round, ((&beta, comm), opening)) in fold_data_iter.enumerate() {
        let log_arity = opening.log_arity as usize;
        let arity = 1 << log_arity;

        // Validate that sibling_values has the expected length (arity - 1)
        if opening.sibling_values.len() != arity - 1 {
            return Err(FriError::SiblingValuesLengthMismatch {
                round,
                expected: arity - 1,
                got: opening.sibling_values.len(),
            });
        }

        // Reconstruct the full evaluation row from self + siblings
        let index_in_group = *start_index % arity;
        let mut evals = vec![EF::ZERO; arity];
        evals[index_in_group] = folded_eval;

        let mut sibling_idx = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..arity {
            if j != index_in_group {
                evals[j] = opening.sibling_values[sibling_idx];
                sibling_idx += 1;
            }
        }

        // Compute the new height after folding
        let log_folded_height = log_current_height - log_arity;

        let dims = &[Dimensions {
            width: arity,
            height: 1 << log_folded_height,
        }];

        // Replace index with the index of the parent FRI node.
        *start_index >>= log_arity;

        // Verify the commitment to the evaluations of the sibling nodes.
        mmcs.verify_batch(
            comm,
            dims,
            *start_index,
            BatchOpeningRef::new(&[evals.clone()], &opening.opening_proof), // It's possible to remove the clone here but unnecessary as evals is tiny.
        )
        .map_err(FriError::CommitPhaseMmcsError)?;

        // Fold the group of sibling nodes to get the evaluation of the parent FRI node.
        folded_eval = folding.fold_row(
            *start_index,
            log_folded_height,
            log_arity,
            beta,
            evals.into_iter(),
        );

        // Update current height
        log_current_height = log_folded_height;

        // If there are new polynomials to roll in at the folded height, do so.
        //
        // Each element of `ro_iter` is the evaluation of a reduced opening polynomial, which is itself
        // a random linear combination `f_{i, 0}(x) + alpha f_{i, 1}(x) + ...`, but when we add it
        // to the current folded polynomial evaluation claim, we need to multiply by a new random factor
        // since `f_{i, 0}` has no leading coefficient.
        //
        // We use `beta^arity` as the random factor to maintain independence.
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            let beta_pow = beta.exp_power_of_2(log_arity);
            folded_eval += beta_pow * ro;
        }
    }

    // Verify we reached the expected final height
    if log_current_height != log_final_height {
        return Err(FriError::FinalFoldHeightMismatch {
            expected: log_final_height,
            got: log_current_height,
        });
    }

    // If ro_iter is not empty, we failed to fold in some polynomial evaluations.
    if let Some((next_log_height, _)) = ro_iter.next() {
        return Err(FriError::UnconsumedReducedOpenings {
            next_log_height,
            remaining: 1 + ro_iter.count(),
        });
    }

    // If we reached this point, we have verified that, starting at the initial index,
    // the chain of folds has produced folded_eval.
    Ok(folded_eval)
}

/// Given an index and a collection of opening proofs, check all opening proofs and combine
/// the opened values into the FRI inputs along the path specified by the index.
///
/// In cases where the maximum height of a batch of matrices is smaller than the
/// global max height, shift the index down to compensate.
///
/// We combine the functions by mapping each function and opening point pair to `(f(z) - f(x))/(z - x)`
/// and then combining functions of the same degree using the challenge alpha.
///
/// ## Arguments:
/// - `params`: The FRI parameters.
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `index`: The index at which to open the functions.
/// - `input_openings`: A vector of batch openings with each opening containing a
///   list of opened values for a collection of matrices along with a batched opening proof.
/// - `alpha`: The challenge used to combine the functions.
/// - `input_mmcs`: The input multi-matrix commitment scheme.
/// - `commitments_with_opening_points`: A vector of joint commitments to collections of matrices
///   and openings of those matrices at a collection of points.
#[inline]
pub(crate) fn open_input<Val, Challenge, InputMmcs, FriMmcs>(
    params: &FriParameters<FriMmcs>,
    log_global_max_height: usize,
    index: usize,
    input_openings: &[BatchOpening<Val, InputMmcs>],
    alpha: Challenge,
    input_mmcs: &InputMmcs,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
) -> Result<FriOpenings<Challenge>, FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
{
    // For each log_height, we store the alpha power and compute the reduced opening.
    // log_height -> (alpha_pow, reduced_opening)
    let mut reduced_openings = BTreeMap::<usize, (Challenge, Challenge)>::new();

    debug_assert_eq!(
        input_openings.len(),
        commitments_with_opening_points.len(),
        "input openings are derived from the same protocol shape as commitments",
    );

    // For each batch commitment and opening.
    for (batch, (batch_opening, (batch_commit, mats))) in input_openings
        .iter()
        .zip(commitments_with_opening_points.iter())
        .enumerate()
    {
        // Find the height of each matrix in the batch.
        // Currently we only check domain.size() as the shift is
        // assumed to always be Val::GENERATOR.
        let batch_dims = mats
            .iter()
            .map(|(domain, points)| Dimensions {
                width: points
                    .iter()
                    .map(|(_, values)| values.len())
                    .all_equal_value()
                    .unwrap(),
                height: domain.size() << params.log_blowup,
            })
            .collect_vec();

        // If the maximum height of the batch is smaller than the global max height,
        // we need to correct the index by right shifting it.
        // If the batch is empty, we set the index to 0.
        let reduced_index = batch_dims
            .iter()
            .map(|dims| dims.height)
            .max()
            .map(|h| index >> (log_global_max_height - log2_strict_usize(h)))
            .unwrap_or(0);

        if batch_opening.opened_values.len() != mats.len() {
            return Err(FriError::BatchOpenedValuesCountMismatch {
                batch,
                expected: mats.len(),
                got: batch_opening.opened_values.len(),
            });
        }

        input_mmcs
            .verify_batch(
                batch_commit,
                &batch_dims,
                reduced_index,
                batch_opening.into(),
            )
            .map_err(FriError::InputError)?;

        // For each matrix in the commitment
        for (matrix, (mat_opening, (mat_domain, mat_points_and_values))) in batch_opening
            .opened_values
            .iter()
            .zip(mats.iter())
            .enumerate()
        {
            let log_height = log2_strict_usize(mat_domain.size()) + params.log_blowup;

            let bits_reduced = log_global_max_height - log_height;
            let rev_reduced_index = reverse_bits_len(index >> bits_reduced, log_height);

            // TODO: this can be nicer with domain methods?

            // Compute gh^i
            let x = Val::GENERATOR
                * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

            let (alpha_pow, ro) = reduced_openings
                .entry(log_height) // Get a mutable reference to the entry.
                .or_insert((Challenge::ONE, Challenge::ZERO));

            // For each polynomial `f` in our matrix, compute `(f(z) - f(x))/(z - x)`,
            // scale by the appropriate alpha power and add to the reduced opening for this log_height.
            for (point, (z, ps_at_z)) in mat_points_and_values.iter().enumerate() {
                let quotient = (*z - x).inverse();
                if mat_opening.len() != ps_at_z.len() {
                    return Err(FriError::PointEvaluationCountMismatch {
                        batch,
                        matrix,
                        point,
                        expected: mat_opening.len(),
                        got: ps_at_z.len(),
                    });
                }
                for (&p_at_x, &p_at_z) in mat_opening.iter().zip(ps_at_z.iter()) {
                    // Note we just checked batch proofs to ensure p_at_x is correct.
                    // x, z were sent by the verifier.
                    // ps_at_z was sent to the verifier and we are using fri to prove it is correct.
                    *ro += *alpha_pow * (p_at_z - p_at_x) * quotient;
                    *alpha_pow *= alpha;
                }
            }
        }

        // `reduced_openings` would have a log_height = log_blowup entry only if there was a
        // trace matrix of height 1. In this case `f` is constant, so `f(zeta) - f(x))/(zeta - x)`
        // must equal `0`.
        if let Some((_, ro)) = reduced_openings.get(&params.log_blowup)
            && !ro.is_zero()
        {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    // Return reduced openings descending by log_height.
    Ok(reduced_openings
        .into_iter()
        .rev()
        .map(|(log_height, (_, ro))| (log_height, ro))
        .collect())
}
