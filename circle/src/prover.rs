use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_challenger::fs::{DefaultCodec, ExtensionFieldCodec, FieldToFieldCodec, ProverState};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{BatchDimensions, Mmcs, MmcsWriter};
use p3_field::{ExtensionField, PrimeField};
use p3_fri::{FriFoldingStrategy, FriParameters, compute_log_arity_for_round};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::protocol::{CircleLabelsDefault, FriLabels};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    transcript: &mut ProverState<Challenger>,
    mut open_input: impl FnMut(usize, usize, &mut ProverState<Challenger>),
) where
    Val: PrimeField,
    Challenge: ExtensionField<Val>,
    M: MmcsWriter<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger<Witness = Val> + DefaultCodec<M::Digest>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    let fri_mmcs = &params.mmcs;
    assert_eq!(
        params.log_final_poly_len, 0,
        "Circle PCS currently requires log_final_poly_len = 0",
    );
    assert!(
        inputs
            .iter()
            .tuple_windows()
            .all(|(l, r)| l.len() >= r.len())
    );

    let log_max_height = log2_strict_usize(inputs[0].len());

    transcript.begin_protocol::<()>(CircleLabelsDefault::PROTOCOL);
    let commit_phase_result = commit_phase(folding, params, inputs, transcript);

    transcript.pow(
        CircleLabelsDefault::pow_query(params.query_proof_of_work_bits),
        params.query_proof_of_work_bits,
    );

    info_span!("query phase").in_scope(|| {
        for query in 0..params.num_queries {
            let index = transcript
                .challenge_bits(
                    CircleLabelsDefault::query_index(query),
                    log_max_height + folding.extra_query_index_bits(),
                )
                .into_inner();
            open_input(query, index, transcript);
            answer_query(
                query,
                fri_mmcs,
                &commit_phase_result.log_arities,
                &commit_phase_result.data,
                index >> folding.extra_query_index_bits(),
                transcript,
            );
        }
    });
    transcript.end_protocol::<()>(CircleLabelsDefault::PROTOCOL);
}

struct CommitPhaseResult<F: Send + Sync + Clone, M: Mmcs<F>> {
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    log_arities: Vec<usize>,
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    transcript: &mut ProverState<Challenger>,
) -> CommitPhaseResult<Challenge, M>
where
    Val: PrimeField,
    Challenge: ExtensionField<Val>,
    M: MmcsWriter<Challenge>,
    Challenger: FieldChallenger<Val> + DefaultCodec<M::Digest>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    let mmcs = &params.mmcs;
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut data = vec![];
    let mut log_arities = vec![];

    // For Circle, we fold down to blowup elements (no separate final_poly_len).
    let log_final_height = params.log_blowup;

    while folded.len() > params.blowup() {
        let round = log_arities.len();
        let log_current_height = log2_strict_usize(folded.len());
        let next_input_log_height = inputs_iter.peek().map(|v| log2_strict_usize(v.len()));
        // Compute the arity for this round
        let log_arity = compute_log_arity_for_round(
            log_current_height,
            next_input_log_height,
            log_final_height,
            params.max_log_arity,
        );
        let arity = 1 << log_arity;
        log_arities.push(log_arity);

        let leaves = RowMajorMatrix::new(folded, arity);
        let dimensions = leaves.dimensions();
        let (commitment, prover_data) = Mmcs::commit(mmcs, vec![leaves]);
        mmcs.write_commitment(
            transcript,
            CircleLabelsDefault::round_commitment(round),
            commitment,
        );

        let beta = transcript
            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                CircleLabelsDefault::beta(round),
            )
            .into_inner();
        debug_assert_eq!(dimensions.width, arity);

        // We passed ownership of `current` to the MMCS, so get a reference to it
        let leaves = mmcs.get_matrices(&prover_data).pop().unwrap();
        folded = folding.fold_matrix(beta, log_arity, leaves.as_view());

        data.push(prover_data);

        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            izip!(&mut folded, v).for_each(|(c, x)| *c += x);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(folded.len(), params.blowup());
    let final_poly = folded[0];
    for x in folded {
        assert_eq!(x, final_poly);
    }
    transcript.add_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
        CircleLabelsDefault::final_poly(),
        &final_poly,
    );

    CommitPhaseResult { data, log_arities }
}

fn answer_query<Val, Challenge, M, Challenger>(
    query: usize,
    mmcs: &M,
    log_arities: &[usize],
    commit_phase_data: &[M::ProverData<RowMajorMatrix<Challenge>>],
    start_index: usize,
    transcript: &mut ProverState<Challenger>,
) where
    Val: PrimeField,
    Challenge: ExtensionField<Val>,
    M: MmcsWriter<Challenge>,
    Challenger: FieldChallenger<Val> + DefaultCodec<M::Digest>,
{
    let mut current_index = start_index;

    for (round, prover_data) in commit_phase_data.iter().enumerate() {
        let matrix = mmcs.get_matrices(prover_data).pop().unwrap();
        let arity = matrix.width();
        let log_arity = log_arities[round];
        debug_assert_eq!(arity, 1 << log_arity);

        // Index of this element within its group
        let index_in_group = current_index % arity;
        // Index of the group (row in the committed matrix)
        let group_index = current_index >> log_arity;

        let (mut opened_rows, opening_proof) = mmcs.open_batch(group_index, prover_data).unpack();
        assert_eq!(opened_rows.len(), 1);
        let mut opened_row = opened_rows.pop().unwrap();
        assert_eq!(opened_row.len(), arity);
        opened_row.remove(index_in_group);

        transcript
            .add_hints::<Challenge, ExtensionFieldCodec<Val, Challenge, FieldToFieldCodec<Val>>>(
                CircleLabelsDefault::sibling_values(query, round),
                &opened_row,
            );
        mmcs.write_proof_hint(
            transcript,
            CircleLabelsDefault::round_opening_proof(query, round),
            &BatchDimensions::single(matrix.dimensions()),
            opening_proof,
        );

        // Update current_index for the next round
        current_index = group_index;
    }
}
