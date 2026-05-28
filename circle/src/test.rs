use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::fs::codecs::decode_field::field_byte_size;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, InteractionPattern, ProverState, VerifierState,
};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::{ExtensionMmcs, Mmcs, MmcsReader, MmcsWriter, Pcs};
use p3_field::extension::BinomialExtensionField;
use p3_fri::FriParameters;
use p3_fri::protocol::Protocol as _;
use p3_fri::verifier::FriError;
use p3_keccak::Keccak256Hash;
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::folding::CircleFriFolding;
use crate::protocol::{
    BatchSpec, CircleLabels, CircleLabelsDefault as Labels, CircleProtocol, FriLabels, MatrixSpec,
};
use crate::{CircleDomain, CirclePcs, InputError};

type Val = Mersenne31;
type Challenge = BinomialExtensionField<Val, 3>;
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
type MyPcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
type Domain = CircleDomain<Val>;
type Proof = Vec<u8>;
type TestError = FriError<
    <ChallengeMmcs as Mmcs<Challenge>>::Error,
    InputError<<ValMmcs as Mmcs<Val>>::Error, <ChallengeMmcs as Mmcs<Challenge>>::Error>,
>;

fn challenger() -> Challenger {
    Challenger::from_hasher(vec![], ByteHash {})
}

fn input_mmcs() -> ValMmcs {
    let byte_hash = ByteHash {};
    let hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
    ValMmcs::new(hash, compress, 0)
}

fn fri_mmcs() -> ChallengeMmcs {
    ChallengeMmcs::new(input_mmcs())
}

fn params() -> FriParameters<ChallengeMmcs> {
    FriParameters::new_testing(fri_mmcs(), 0)
}

fn domain_separator(
    protocol: &CircleProtocol,
    params: &FriParameters<ChallengeMmcs>,
    input_mmcs: &ValMmcs,
) -> DomainSeparator<u8> {
    let mut interactions = Vec::new();
    protocol.append_pcs_interactions::<Val, Challenge, ValMmcs, ChallengeMmcs>(
        &mut interactions,
        input_mmcs,
        params,
    );
    protocol.append_fri_pattern::<Val, Challenge, ValMmcs, ChallengeMmcs>(
        &mut interactions,
        params,
        input_mmcs,
    );
    let mut ds = DomainSeparator::new(
        1,
        b"circle-pcs-transcript-test",
        InteractionPattern::new(interactions).unwrap(),
    );
    ds.bind_pattern_hash();
    ds
}

fn minimal_batch() -> BatchSpec {
    const LOG_N: usize = 10;
    const WIDTH: usize = 1;

    BatchSpec::new(vec![MatrixSpec::new(
        Dimensions {
            width: WIDTH,
            height: 1 << LOG_N,
        },
        1,
    )])
}

fn rand_matrix<R: RngExt>(
    rng: &mut R,
    spec: &BatchSpec,
    pcs: &MyPcs,
) -> Vec<(Domain, RowMajorMatrix<Val>)> {
    spec.matrices()
        .iter()
        .map(|matrix_spec| {
            let dimensions = matrix_spec.dimensions();
            (
                <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                    pcs,
                    dimensions.height,
                ),
                RowMajorMatrix::<Val>::rand(rng, dimensions.height, dimensions.width),
            )
        })
        .collect::<Vec<_>>()
}

fn run_prover<R: RngExt>(
    rng: &mut R,
    pcs: &MyPcs,
    protocol: &CircleProtocol,
    transcript: &mut ProverState<Challenger>,
) {
    let matrices = protocol
        .batches
        .iter()
        .map(|batch| rand_matrix(rng, batch, pcs))
        .collect::<Vec<_>>();

    let (commitments, data) = matrices
        .into_iter()
        .map(|matrices| <MyPcs as Pcs<Challenge, Challenger>>::commit(pcs, matrices))
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let mut opening_points = Vec::with_capacity(protocol.batches.len());
    for (batch_index, (commitment, batch)) in
        commitments.into_iter().zip(&protocol.batches).enumerate()
    {
        pcs.mmcs
            .write_commitment(transcript, Labels::commitment(batch_index), commitment);

        let batch_opening_points = batch
            .matrices()
            .iter()
            .enumerate()
            .map(|(matrix_index, matrix)| {
                (0..matrix.num_opening_points())
                    .map(|point_index| {
                        transcript
                            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                                Labels::opening_point(batch_index, matrix_index, point_index),
                            )
                            .into_inner()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        opening_points.push(batch_opening_points);
    }

    <MyPcs as Pcs<Challenge, Challenger>>::open(
        pcs,
        data.iter().zip(opening_points).collect::<Vec<_>>(),
        transcript,
    );
}

fn run_verifier(
    pcs: &MyPcs,
    protocol: &CircleProtocol,
    transcript: &mut VerifierState<'_, Challenger>,
) -> Result<(), TestError> {
    let mut commitments = Vec::with_capacity(protocol.batches.len());
    let mut opening_points = Vec::with_capacity(protocol.batches.len());
    for (batch_index, batch) in protocol.batches.iter().enumerate() {
        let commitment = pcs
            .mmcs
            .read_commitment(
                transcript,
                Labels::commitment(batch_index),
                &protocol.input_batch_dimensions[batch_index],
            )?
            .into_inner();
        commitments.push(commitment);

        let points = batch
            .matrices()
            .iter()
            .enumerate()
            .map(|(matrix_index, matrix)| {
                (0..matrix.num_opening_points())
                    .map(|point_index| {
                        transcript
                            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                                Labels::opening_point(batch_index, matrix_index, point_index),
                            )
                            .into_inner()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        opening_points.push(points);
    }

    let commitments_with_opening_points = protocol
        .batches
        .iter()
        .zip(commitments.into_iter())
        .enumerate()
        .map(|(batch_index, (batch, commitment))| {
            let batch_opening_points = batch
                .matrices()
                .iter()
                .enumerate()
                .map(|(matrix_index, matrix)| {
                    let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                        pcs,
                        matrix.height(),
                    );
                    let points_and_values = opening_points[batch_index][matrix_index]
                        .iter()
                        .enumerate()
                        .map(|(point_index, &point)| {
                            let values = transcript
                                .next_extensions::<Val, Challenge, FieldToFieldCodec<Val>>(
                                    Labels::opened_values(batch_index, matrix_index, point_index),
                                    matrix.width(),
                                )?
                                .into_iter()
                                .map(|value| value.into_inner())
                                .collect::<Vec<_>>();
                            Ok((point, values))
                        })
                        .collect::<Result<Vec<_>, TestError>>()?;
                    Ok((domain, points_and_values))
                })
                .collect::<Result<Vec<_>, TestError>>()?;
            Ok((commitment, batch_opening_points))
        })
        .collect::<Result<Vec<_>, TestError>>()?;

    <MyPcs as Pcs<Challenge, Challenger>>::verify(pcs, commitments_with_opening_points, transcript)
}

fn prover<R: RngExt>(rng: &mut R, pcs: &MyPcs, protocol: &CircleProtocol) -> Proof {
    let ds = domain_separator(protocol, &pcs.fri_params, &pcs.mmcs);
    let mut transcript = ProverState::<_, u8>::new(challenger(), &ds);
    run_prover(rng, pcs, protocol, &mut transcript);
    transcript.finalize()
}

fn verifier(pcs: &MyPcs, protocol: &CircleProtocol, proof: Proof) -> Result<(), TestError> {
    let ds = domain_separator(protocol, &pcs.fri_params, &pcs.mmcs);
    let mut transcript = VerifierState::<_, u8>::new(challenger(), &ds, &proof);
    run_verifier(pcs, protocol, &mut transcript)?;
    transcript.finalize().map_err(FriError::Transcript)
}

fn minimal_proof() -> (MyPcs, CircleProtocol, DomainSeparator<u8>, Proof) {
    let mut rng = SmallRng::seed_from_u64(0);
    let input_mmcs = input_mmcs();
    let pcs = MyPcs::new(input_mmcs, params());
    let protocol = CircleProtocol::new(
        &pcs.fri_params,
        vec![minimal_batch()],
        CircleFriFolding::extra_query_index_bits(),
    );
    let ds = domain_separator(&protocol, &pcs.fri_params, &pcs.mmcs);
    let proof = prover(&mut rng, &pcs, &protocol);
    (pcs, protocol, ds, proof)
}

fn tamper_field_at_label(ds: &DomainSeparator<u8>, label: &'static str, proof: &Proof) -> Proof {
    let mut proof = proof.clone();
    let offset = ds
        .byte_offset_for_label::<Val, Challenge, FieldToFieldCodec<Val>, Challenger>(label)
        .expect("label should carry proof bytes");
    proof[offset + field_byte_size::<Val>() - 1] ^= 1;
    proof
}

fn tamper_pow_at_label(ds: &DomainSeparator<u8>, label: &'static str, proof: &Proof) -> Proof {
    let mut proof = proof.clone();
    let offset = ds
        .byte_offset_for_label::<Val, Challenge, FieldToFieldCodec<Val>, Challenger>(label)
        .expect("PoW label should carry proof bytes");
    assert!(offset + field_byte_size::<Val>() <= proof.len());
    proof[offset] |= 0x80;
    proof
}

#[test]
fn test_minimal_fixture() {
    let (pcs, protocol, _ds, proof) = minimal_proof();
    verifier(&pcs, &protocol, proof).unwrap();
}

#[test]
fn tampered_opened_values_rejects_verification() {
    let (pcs, protocol, ds, proof) = minimal_proof();
    let proof = tamper_field_at_label(&ds, Labels::opened_values(0, 0, 0), &proof);
    verifier(&pcs, &protocol, proof).expect_err("tampered opened values should fail");
}

#[test]
fn tampered_lambda_rejects_verification() {
    let (pcs, protocol, ds, proof) = minimal_proof();
    let proof = tamper_field_at_label(&ds, Labels::lambdas(), &proof);
    verifier(&pcs, &protocol, proof).expect_err("tampered lambda should fail");
}

#[test]
fn tampered_input_opening_proof_hits_input_error() {
    let (pcs, protocol, ds, proof) = minimal_proof();
    let proof = tamper_field_at_label(&ds, Labels::input_opening_proof(0, 0), &proof);
    let err = verifier(&pcs, &protocol, proof).expect_err("tampered input proof should fail");

    match err {
        FriError::InputError(InputError::InputMmcsError(_)) => {}
        other => panic!("wrong error variant: {other:?}"),
    }
}

#[test]
fn tampered_first_layer_proof_hits_input_error() {
    let (pcs, protocol, ds, proof) = minimal_proof();
    let proof = tamper_field_at_label(&ds, Labels::first_layer_opening_proof(0), &proof);
    let err = verifier(&pcs, &protocol, proof).expect_err("tampered first-layer proof should fail");

    match err {
        FriError::InputError(InputError::FirstLayerMmcsError(_)) => {}
        other => panic!("wrong error variant: {other:?}"),
    }
}

#[test]
fn tampered_circle_fri_round_opening_hits_commit_phase_mmcs_error() {
    let (pcs, protocol, ds, proof) = minimal_proof();
    let proof = tamper_field_at_label(&ds, Labels::round_opening_proof(0, 0), &proof);
    let err = verifier(&pcs, &protocol, proof).expect_err("tampered FRI round proof should fail");

    match err {
        FriError::CommitPhaseMmcsError(_) => {}
        other => panic!("wrong error variant: {other:?}"),
    }
}

#[test]
fn tampered_query_pow_hits_transcript_error() {
    let (pcs, protocol, ds, proof) = minimal_proof();
    let proof = tamper_pow_at_label(
        &ds,
        Labels::pow_query(pcs.fri_params.query_proof_of_work_bits),
        &proof,
    );
    let err = verifier(&pcs, &protocol, proof).expect_err("tampered query PoW should fail");

    match err {
        FriError::Transcript(_) => {}
        other => panic!("wrong error variant: {other:?}"),
    }
}

#[test]
fn truncated_proof_hits_transcript_error() {
    let (pcs, protocol, _ds, mut proof) = minimal_proof();
    proof.pop();
    let err = verifier(&pcs, &protocol, proof).expect_err("truncated proof should fail");

    match err {
        FriError::Transcript(_) => {}
        other => panic!("wrong error variant: {other:?}"),
    }
}
