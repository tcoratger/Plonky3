use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_challenger::fs::{DefaultCodec, ExtensionFieldCodec, FieldToFieldCodec, ProverState};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{BatchDimensions, Mmcs, MmcsWriter};
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, PrimeField, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::{debug_span, info_span, instrument};

use crate::protocol::{FriLabels, FriLabelsDefault};
use crate::{
    FriFoldingStrategy, FriParameters, ProverDataWithOpeningPoints, compute_log_arity_for_round,
};

/// Create a proof that an opening `f(zeta)` is correct by proving that the
/// function `(f(x) - f(zeta))/(x - zeta)` is low degree.
///
/// This further supports proving a batch of these claims for a collection of polynomials of shrinking degrees.
/// Polynomials of equal degree can be combined using randomness before calling this function.
///
/// The Soundness error from prove_fri comes from the paper:
/// Proximity Gaps for Reed-Solomon Codes (https://eprint.iacr.org/2020/654)
/// and is either `rate^{num_queries}` or `rate^{num_queries/2}` depending on if you rely on conjectured or
/// proven soundness. Particularly safety conscious users may want to set `num_queries` slightly higher than
/// this to account for the fact that most implementations batch inputs using a single random challenge
/// instead of one challenge for each polynomial and due to the birthday paradox,
/// there is a non trivial chance that two queried indices will be equal.
///
/// Arguments:
/// - `folding`: The FRI folding scheme to use.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `inputs`: The evaluation vectors of all polynomials we are applying FRI to. The function assumes that
///   commitments to these vectors have been produced and observed by the challenger earlier in the protocol.
/// - `transcript`: The Fiat-Shamir prover transcript used to write proof data and sample challenges.
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `prover_data_with_opening_points`: A list of pairs of a batch commitment to a collection
///   of matrices and a list of points to open those matrices at.
#[instrument(name = "FRI prover", skip_all)]
pub fn prove_fri<Folding, Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    inputs: Vec<Vec<Challenge>>,
    transcript: &mut ProverState<Challenger>,
    log_global_max_height: usize,
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Val>>,
    >],
    input_mmcs: &InputMmcs,
) where
    Val: TwoAdicField + PrimeField,
    Challenge: ExtensionField<Val>,
    InputMmcs: MmcsWriter<Val>,
    FriMmcs: MmcsWriter<Challenge>,
    Challenger: FieldChallenger<Val>
        + GrindingChallenger<Witness = Val>
        + DefaultCodec<InputMmcs::Digest>
        + DefaultCodec<FriMmcs::Digest>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    assert!(!inputs.is_empty());
    assert!(
        inputs
            .iter()
            .tuple_windows()
            .all(|(l, r)| l.len() >= r.len()),
        "Inputs are not sorted in descending order of length."
    );

    let log_max_height = log2_strict_usize(inputs[0].len());
    let log_min_height = log2_strict_usize(inputs.last().unwrap().len());
    if params.log_final_poly_len > 0 {
        // Final_poly_degree must be less than or equal to the degree of the smallest polynomial.
        assert!(log_min_height > params.log_final_poly_len + params.log_blowup);
    }

    transcript.begin_protocol::<()>(FriLabelsDefault::PROTOCOL);

    // Continually fold the inputs down until the polynomial degree reaches final_poly_degree.
    // Returns a vector of commitments to the intermediate stage polynomials, the intermediate stage polynomials
    // themselves and the final polynomial.
    // Note that the challenger observes the commitments and the final polynomial inside this function so we don't
    // need to observe the output of this function here.
    let commit_phase_data = commit_phase::<_, _, _, _, _>(folding, params, inputs, transcript);
    // Produce a proof of work witness before receiving any query challenges.
    // This helps to prevent grinding attacks.
    transcript.pow(
        FriLabelsDefault::pow_query(params.query_proof_of_work_bits),
        params.query_proof_of_work_bits,
    );

    info_span!("query phase").in_scope(|| {
        // Sample num_queries indexes to check.
        // The probability that no two FRI indices are equal (ignoring extra query index bits) is:
        // (Grabbed this from wikipedia page on the birthday problem)
        // N!/(N^{num_queries} * (N - num_queries)!) ~ (1 - 1/N)^{num_queries * (num_queries - 1)/2}
        //                                           ~ (1 - num_queries^2/2N)
        // Here N = 2^log_max_height.
        // With num_queries = 100, N = 2^20, this is 0.995 so there is a .5% chance of a collision.
        // Due to this, security conscious users may want to set num_queries a little higher than the
        // theoretical minimum.
        for query in 0..params.num_queries {
            let index = transcript
                .challenge_bits(
                    FriLabelsDefault::query_index(query),
                    log_max_height + folding.extra_query_index_bits(),
                )
                .into_inner();
            // For each index, create a proof that the folding operations along the chain are correct.
            // With variable arity, the index shifts by log_arity each round.
            open_input::<_, _, _, _>(
                query,
                log_global_max_height,
                index,
                prover_data_with_opening_points,
                input_mmcs,
                transcript,
            );
            answer_query::<_, _, _, _>(
                query,
                &params.mmcs,
                &commit_phase_data,
                index >> folding.extra_query_index_bits(),
                transcript,
            );
        }
    });
    transcript.end_protocol::<()>(FriLabelsDefault::PROTOCOL);
}

/// Perform the commit phase of the FRI protocol.
///
/// In each round we reduce our evaluations over `H` to evaluations over `H^k` (where k is the arity)
/// by folding with a random challenge. For instance, for arity 2, we have:
/// ```text
///     f_{i + 1}(x^2) = (f_i(x) + f_i(-x))/2 + beta_i (f_i(x) - f_i(-x))/2x
/// ```
/// We then commit to the evaluation vector over the smaller domain, i.e. `f_{i + 1}` over `H^2` for arity 2.
///
/// The arity for each round is dynamically computed to ensure we always commit at each input height level.
///
/// Once the degree of our polynomial falls below `final_poly_degree`, we compute the coefficients of our
/// polynomial and return them along with all intermediate evaluations and corresponding commitments.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `inputs`: The evaluation vectors of the polynomials. These must be sorted in descending order of length and each
///   evaluation vector must be in bit reversed order. This function assumes that commitments to these vectors
///   have already been produced and observed by the challenger.
/// - `transcript`: The Fiat-Shamir prover transcript used to write proof data and sample challenges.
#[instrument(skip_all)]
fn commit_phase<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    transcript: &mut ProverState<Challenger>,
) -> Vec<M::ProverData<RowMajorMatrix<Challenge>>>
where
    Val: TwoAdicField + PrimeField,
    Challenge: ExtensionField<Val>,
    M: MmcsWriter<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger<Witness = Val> + DefaultCodec<M::Digest>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut data = vec![];
    let mut log_arities = vec![];

    let log_final_height = params.log_blowup + params.log_final_poly_len;

    let mut round = 0;
    while folded.len() > params.blowup() * params.final_poly_len() {
        let log_current_height = log2_strict_usize(folded.len());
        let next_input_log_height = inputs_iter.peek().map(|v| log2_strict_usize(v.len()));

        // Compute the arity for this round.
        let log_arity = compute_log_arity_for_round(
            log_current_height,
            next_input_log_height,
            log_final_height,
            params.max_log_arity,
        );
        let arity = 1 << log_arity;
        log_arities.push(log_arity);

        // As folded is in bit reversed order, the evaluations at conjugate points are adjacent.
        // We reinterpret the vector as a matrix of width `arity`.
        let leaves = RowMajorMatrix::new(folded, arity);

        // Commit to these evaluations and observe the commitment.
        let (commitment, prover_data) = Mmcs::commit(&params.mmcs, vec![leaves]);
        params.mmcs.write_commitment(
            transcript,
            FriLabelsDefault::round_commitment(round),
            commitment,
        );

        // Produce a proof of work witness after observing the commitment and
        // before the Fiat-Shamir batching challenge.
        transcript.pow(
            FriLabelsDefault::pow_round(params.commit_proof_of_work_bits, round),
            params.commit_proof_of_work_bits,
        );

        // Get the Fiat-Shamir challenge for this round.
        let beta = transcript
            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(FriLabelsDefault::beta(
                round,
            ))
            .into_inner();
        round += 1;

        // We passed ownership of `leaves` to the MMCS, so get a reference to it.
        let leaves = params.mmcs.get_matrices(&prover_data).pop().unwrap();
        // Do the folding operation with the computed arity.
        folded = folding.fold_matrix(beta, log_arity, leaves.as_view());

        data.push(prover_data);

        // If we have reached the size of the next input vector, we can add it to the current vector.
        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            // Each element of `inputs_iter` is a reduced opening polynomial, which is itself a
            // random linear combination `f_{i, 0} + alpha f_{i, 1} + ...`, when we add it
            // to the current folded polynomial, we need to multiply by a random factor.
            // We use beta^arity as the random factor to maintain independence.
            let beta_pow = beta.exp_power_of_2(log_arity);
            izip!(&mut folded, v).for_each(|(c, x)| *c += beta_pow * x);
        }
    }

    // Now we need to get the coefficients of the final polynomial. As we know that the degree
    // is `<= params.final_poly_len()` and the evaluations are stored in bit-reversed order,
    // we can just truncate the folded vector, bit-reverse again and run an IDFT.
    folded.truncate(params.final_poly_len());
    reverse_slice_index_bits(&mut folded);
    let final_poly = debug_span!("idft final poly")
        .in_scope(|| Radix2DFTSmallBatch::default().idft_algebra(folded));

    // Write all coefficients of the final polynomial.
    transcript.add_extensions::<Val, Challenge, FieldToFieldCodec<Val>>(
        FriLabelsDefault::final_poly(),
        &final_poly,
    );

    // Bind the chosen folding arities into the transcript.
    let log_arity_values = log_arities
        .iter()
        .map(|&log_arity| Val::from_usize(log_arity))
        .collect::<Vec<_>>();

    transcript.add_scalars::<Val, FieldToFieldCodec<Val>>(
        FriLabelsDefault::log_arities(),
        &log_arity_values,
    );

    data
}

/// Given an `index` produce a proof that the chain of folds are correct.
/// This is the prover's complement to the verifier's [`verify_query`] function.
///
/// In addition to the output of this function, the prover must also supply the verifier with the input values
/// (with associated opening proofs). These are produced by the `open_input` function passed into `prove_fri`.
///
/// For each round `i`, this returns the sibling values (all values in the group except the queried one)
/// along with an opening proof. The verifier can then reconstruct the full group, verify the commitment,
/// and fold to get the value at the parent index.
///
/// With variable arity, the index shifts by `log_arities[i]` each round instead of always by 1
/// (i.e. when arity is fixed to 2).
///
/// Arguments:
/// - `query`: The FRI query number.
/// - `mmcs`: The commitment scheme for FRI fold commitments.
/// - `folded_polynomial_commits`: A slice of commitments to the intermediate stage polynomials.
/// - `start_index`: The opening index for the unfolded polynomial.
/// - `transcript`: The Fiat-Shamir prover transcript
#[inline]
fn answer_query<Val, F, M, Challenger>(
    query: usize,
    mmcs: &M,
    folded_polynomial_commits: &[M::ProverData<RowMajorMatrix<F>>],
    start_index: usize,
    transcript: &mut ProverState<Challenger>,
) where
    Val: PrimeField,
    F: ExtensionField<Val>,
    M: MmcsWriter<F>,
    Challenger: FieldChallenger<Val> + DefaultCodec<M::Digest>,
{
    let mut current_index = start_index;

    for (round, prover_data) in folded_polynomial_commits.iter().enumerate() {
        let matrix = mmcs.get_matrices(prover_data).pop().unwrap();
        let arity = matrix.width();
        let log_arity = log2_strict_usize(arity);

        // Index of this element within its group of `arity` elements.
        let index_in_group = current_index % arity;
        // Index of the group, i.e. the row in the committed matrix.
        let group_index = current_index >> log_arity;

        // Get a proof that the group of evaluations is correct.
        let (mut opened_rows, opening_proof) =
            Mmcs::open_batch(mmcs, group_index, prover_data).unpack();

        // opened_rows should contain just the values in this group.
        assert_eq!(opened_rows.len(), 1);
        let mut opened_row = opened_rows.pop().unwrap();
        assert_eq!(
            opened_row.len(),
            arity,
            "Committed data should have arity {} elements",
            arity
        );

        // Write all siblings, excluding the queried value, along with the opening proof.
        opened_row.remove(index_in_group);
        transcript.add_hints::<F, ExtensionFieldCodec<Val, F, FieldToFieldCodec<Val>>>(
            FriLabelsDefault::sibling_values(query, round),
            &opened_row,
        );
        mmcs.write_proof_hint(
            transcript,
            FriLabelsDefault::round_opening_proof(query, round),
            &BatchDimensions::single(matrix.dimensions()),
            opening_proof,
        );

        current_index = group_index;
    }
}

/// Given an index, produce batch opening proofs for each collection of matrices
/// combined into a single mmcs commitment.
///
/// In cases where the maximum height of a batch of matrices is smaller than the
/// global max height, shift the index down to compensate.
///
/// Arguments:
/// - `query`: The FRI query number.
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `index`: The index to open the matrices at.
/// - `prover_data_with_opening_points`: A list of pairs of a batch commitment to a collection
///   of matrices and a list of points to open those matrices at.
/// - `mmcs`: The mixed matrix commitment scheme used to produce the batch commitments.
fn open_input<Val, Challenge, InputMmcs, Challenger>(
    query: usize,
    log_global_max_height: usize,
    index: usize,
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Val>>,
    >],
    mmcs: &InputMmcs,
    transcript: &mut ProverState<Challenger>,
) where
    Val: TwoAdicField + PrimeField,
    Challenge: ExtensionField<Val>,
    InputMmcs: MmcsWriter<Val>,
    Challenger: FieldChallenger<Val> + DefaultCodec<InputMmcs::Digest>,
{
    // This gives the verifier access to evaluations `f(x)` from which it can compute
    // `(f(zeta) - f(x))/(zeta - x)` and then combine them together and roll into FRI
    // as appropriate.
    for (batch, (data, _)) in prover_data_with_opening_points.iter().enumerate() {
        let log_max_height = log2_strict_usize(mmcs.get_max_height(data));
        let bits_reduced = log_global_max_height - log_max_height;
        // If a matrix is smaller than global max height, we roll it into
        // fri in a later round.
        let reduced_index = index >> bits_reduced;
        let (opened_values, opening_proof) = Mmcs::open_batch(mmcs, reduced_index, data).unpack();
        let dimensions = mmcs
            .get_matrices(data)
            .iter()
            .map(|matrix| matrix.dimensions())
            .collect::<Vec<_>>();
        let dimensions = BatchDimensions::from(dimensions);
        assert_eq!(opened_values.len(), dimensions.len());
        for (matrix, (opened_values, dimensions)) in
            opened_values.iter().zip(dimensions.iter()).enumerate()
        {
            assert_eq!(opened_values.len(), dimensions.width);
            transcript.add_hints::<Val, FieldToFieldCodec<Val>>(
                FriLabelsDefault::input_opened_values(query, batch, matrix),
                opened_values,
            );
        }
        mmcs.write_proof_hint(
            transcript,
            FriLabelsDefault::input_opening_proof(query, batch),
            &dimensions,
            opening_proof,
        );
    }
}
