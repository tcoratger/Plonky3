use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, InteractionPattern, ProverState, VerifierState,
};
use p3_commit::{ExtensionMmcs, Mmcs, MmcsReader, MmcsTranscript, MmcsWriter, Pcs};
use p3_dft::Radix2Dit;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::protocol::{
    BatchSpec, FriLabels, FriLabelsDefault as Labels, FriProtocol, MatrixSpec, Protocol as _,
};
use crate::verifier::FriError;
use crate::{FriParameters, HidingFriPcs, TwoAdicFriFolding, TwoAdicFriPcs};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Folding = TwoAdicFriFolding;
type Proof = Vec<u8>;

pub fn domain_separator<InputMmcs, FriMmcs>(
    protocol: &FriProtocol,
    params: &FriParameters<FriMmcs>,
    input_mmcs: &InputMmcs,
) -> DomainSeparator<u8>
where
    InputMmcs: MmcsTranscript<Val>,
    FriMmcs: MmcsTranscript<Challenge, Digest = Val>,
    InputMmcs::Error: core::fmt::Debug,
{
    let mut interactions = Vec::new();
    protocol.append_pcs_interactions::<Val, Challenge, InputMmcs, FriMmcs>(
        &mut interactions,
        input_mmcs,
        params,
    );
    protocol.append_fri_pattern::<Val, Challenge, InputMmcs, FriMmcs>(
        &mut interactions,
        params,
        input_mmcs,
    );
    let mut ds: DomainSeparator<u8> = DomainSeparator::new(
        1,
        b"fri-test",
        InteractionPattern::new(interactions).unwrap(),
    );
    ds.bind_pattern_hash();
    ds
}

fn random_batches<R: RngExt>(rng: &mut R) -> Vec<BatchSpec> {
    const BATCH_COUNT: usize = 5;
    const MATRICES_PER_BATCH_RANGE: core::ops::Range<usize> = 1..4;
    const LOG_DOMAIN_SIZE_RANGE: core::ops::Range<usize> = 2..6;
    const MATRIX_WIDTH_RANGE: core::ops::Range<usize> = 1..5;
    const OPENING_POINTS_RANGE: core::ops::Range<usize> = 1..4;

    (0..BATCH_COUNT)
        .map(|_| {
            let log_domain_sizes = (0..rng.random_range(MATRICES_PER_BATCH_RANGE))
                .map(|_| rng.random_range(LOG_DOMAIN_SIZE_RANGE))
                .collect::<Vec<_>>();
            BatchSpec::random(
                rng,
                &log_domain_sizes,
                MATRIX_WIDTH_RANGE,
                OPENING_POINTS_RANGE,
            )
        })
        .collect::<Vec<_>>()
}

fn minimal_batch() -> BatchSpec {
    BatchSpec::new(vec![MatrixSpec::new(
        Dimensions {
            width: 2,
            height: 8,
        },
        1,
    )])
}

fn perm() -> Perm {
    const PERM_SEED: u64 = 100;
    let mut rng = SmallRng::seed_from_u64(PERM_SEED);
    Perm::new_from_rng_128(&mut rng)
}

fn challenger() -> Challenger {
    let perm = perm();
    Challenger::new(perm)
}

mod two_adic {
    use p3_challenger::fs::codecs::decode_field::field_byte_size;

    use super::*;
    use crate::verifier::{FriOpenings, verify_query};

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type MyPcs = TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs>;
    type Domain = <MyPcs as Pcs<Challenge, Challenger>>::Domain;
    type TestError = FriError<
        <ChallengeMmcs as p3_commit::Mmcs<Challenge>>::Error,
        <ValMmcs as p3_commit::Mmcs<Val>>::Error,
    >;

    fn fri_mmcs() -> ChallengeMmcs {
        let perm = perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        ChallengeMmcs::new(ValMmcs::new(hash.clone(), compress.clone(), 0))
    }

    fn input_mmcs() -> ValMmcs {
        let perm = perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        ValMmcs::new(hash.clone(), compress.clone(), 0)
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
                    RowMajorMatrix::<Val>::rand_nonzero(rng, dimensions.height, dimensions.width),
                )
            })
            .collect::<Vec<_>>()
    }

    fn run_prover<R: RngExt>(
        rng: &mut R,
        pcs: &MyPcs,
        protocol: &FriProtocol,
        transcript: &mut ProverState<Challenger>,
    ) {
        let matrices = protocol
            .batches
            .iter()
            .map(|batch| rand_matrix(rng, batch, pcs))
            .collect::<Vec<_>>();

        let (commitments, data) = matrices
            .into_iter()
            .map(|mat| <MyPcs as Pcs<Challenge, Challenger>>::commit(pcs, mat))
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let mut opening_points = Vec::with_capacity(protocol.batches.len());
        for (batch_index, (commitment, batch)) in
            commitments.into_iter().zip(&protocol.batches).enumerate()
        {
            pcs.input_mmcs().write_commitment(
                transcript,
                Labels::commitment(batch_index),
                commitment,
            );

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

        pcs.open(
            data.iter().zip(opening_points).collect::<Vec<_>>(),
            transcript,
        );
    }

    fn run_verifier(
        pcs: &MyPcs,
        protocol: &FriProtocol,
        transcript: &mut VerifierState<'_, Challenger>,
    ) -> Result<(), TestError> {
        let mut commitments = Vec::with_capacity(protocol.batches.len());
        let mut opening_points = Vec::with_capacity(protocol.batches.len());
        for (batch_index, batch) in protocol.batches.iter().enumerate() {
            let commitment = pcs
                .input_mmcs()
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
                        let domain =
                            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                                pcs,
                                matrix.height(),
                            );
                        let points_and_values = opening_points[batch_index][matrix_index]
                            .iter()
                            .enumerate()
                            .map(|(point_index, &point)| {
                                let values = transcript
                                    .next_extensions::<Val, Challenge, FieldToFieldCodec<Val>>(
                                        Labels::opened_values(
                                            batch_index,
                                            matrix_index,
                                            point_index,
                                        ),
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

        pcs.verify(commitments_with_opening_points, transcript)
    }

    fn prover<R: RngExt>(rng: &mut R, pcs: &MyPcs, protocol: &FriProtocol) -> Proof {
        let ds = domain_separator(&protocol, pcs.params(), pcs.input_mmcs());
        let mut transcript = ProverState::<_, u8>::new(challenger(), &ds);
        run_prover(rng, pcs, &protocol, &mut transcript);
        transcript.finalize()
    }

    fn verifier(pcs: &MyPcs, protocol: &FriProtocol, proof: Proof) -> Result<(), TestError> {
        let ds = domain_separator(&protocol, pcs.params(), pcs.input_mmcs());
        let mut transcript = VerifierState::<_, u8>::new(challenger(), &ds, &proof);
        run_verifier(pcs, &protocol, &mut transcript)?;
        transcript.finalize().map_err(FriError::Transcript)
    }

    fn run_rand_test(params: FriParameters<ChallengeMmcs>) {
        let mut rng = SmallRng::seed_from_u64(11);
        let input_mmcs = input_mmcs();
        let pcs = MyPcs::new(Radix2Dit::default(), input_mmcs, params);

        for _ in 0..100 {
            let batches = random_batches(&mut rng);
            let protocol = FriProtocol::new(&pcs.fri, batches, Folding::extra_query_index_bits());
            let proof = prover(&mut rng, &pcs, &protocol);
            verifier(&pcs, &protocol, proof).unwrap();
        }
    }

    fn minimal_proof() -> (MyPcs, FriProtocol, DomainSeparator<u8>, Proof) {
        let params = FriParameters::new_testing(fri_mmcs(), 0);
        let mut rng = SmallRng::seed_from_u64(11);
        let input_mmcs = input_mmcs();
        let pcs = MyPcs::new(Radix2Dit::default(), input_mmcs, params);

        let batches = vec![minimal_batch()];
        let protocol = FriProtocol::new(&pcs.fri, batches, Folding::extra_query_index_bits());
        let ds = domain_separator(&protocol, pcs.params(), pcs.input_mmcs());
        let proof = prover(&mut rng, &pcs, &protocol);
        (pcs, protocol, ds, proof)
    }

    fn log_final_height(params: &FriParameters<ChallengeMmcs>) -> usize {
        params.log_blowup + params.log_final_poly_len
    }

    #[test]
    fn test_rand_fixtures() {
        let params = FriParameters {
            mmcs: fri_mmcs(),
            log_blowup: 2,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 5,
            query_proof_of_work_bits: 5,
        };
        run_rand_test(params);
    }

    #[test]
    fn test_minimal_fixture() {
        let params = FriParameters::new_testing(fri_mmcs(), 0);
        let mut rng = SmallRng::seed_from_u64(11);
        let input_mmcs = input_mmcs();
        let pcs = MyPcs::new(Radix2Dit::default(), input_mmcs, params);

        let batches = vec![minimal_batch()];
        let protocol = FriProtocol::new(&pcs.fri, batches, Folding::extra_query_index_bits());
        let proof = prover(&mut rng, &pcs, &protocol);
        verifier(&pcs, &protocol, proof).unwrap();
    }

    #[test]
    fn tampered_input_opening_proof_hits_input_error() {
        let (pcs, protocol, ds, proof) = minimal_proof();
        let proof = tamper_field_at_label(&ds, Labels::input_opening_proof(0, 0), &proof);
        let err =
            verifier(&pcs, &protocol, proof).expect_err("tampered input opening proof should fail");

        match err {
            FriError::InputError(_) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn tampered_round_opening_proof_hits_commit_phase_mmcs_error() {
        let (pcs, protocol, ds, proof) = minimal_proof();
        let proof = tamper_field_at_label(&ds, Labels::round_opening_proof(0, 0), &proof);
        let err = verifier(&pcs, &protocol, proof)
            .expect_err("tampered FRI round opening proof should fail");

        match err {
            FriError::CommitPhaseMmcsError(_) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn tampered_round_pow_hits_transcript_error() {
        let (pcs, protocol, ds, proof) = minimal_proof();
        assert!(!protocol.rounds.dimensions.is_empty());

        let proof = tamper_pow_at_label(
            &ds,
            Labels::pow_round(pcs.params().commit_proof_of_work_bits, 0),
            &proof,
        );
        let err = verifier(&pcs, &protocol, proof).expect_err("tampered round PoW should fail");

        match err {
            FriError::Transcript(_) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn tampered_query_pow_hits_transcript_error() {
        let (pcs, protocol, ds, proof) = minimal_proof();
        let proof = tamper_pow_at_label(
            &ds,
            Labels::pow_query(pcs.params().query_proof_of_work_bits),
            &proof,
        );
        let err = verifier(&pcs, &protocol, proof).expect_err("tampered query PoW should fail");

        match err {
            FriError::Transcript(_) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn tampered_opened_values_rejects_verification() {
        let (pcs, protocol, ds, proof) = minimal_proof();
        let proof = tamper_field_at_label(&ds, Labels::opened_values(0, 0, 0), &proof);
        let err = verifier(&pcs, &protocol, proof).expect_err("tampered opened values should fail");

        match err {
            FriError::FinalPolyMismatch
            | FriError::CommitPhaseMmcsError(_)
            | FriError::InputError(_)
            | FriError::Transcript(_) => {}
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

    #[test]
    fn missing_initial_reduced_opening_hits_fri_error() {
        let (pcs, protocol, _ds, _proof) = minimal_proof();
        let folding = TwoAdicFriFolding;
        let reduced_openings: FriOpenings<Challenge> = vec![];
        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<crate::CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data_iter = betas.iter().zip(commits.iter()).zip(openings.iter());
        let mut start_index = 0;

        let err: TestError = verify_query::<
            TwoAdicFriFolding,
            Val,
            Challenge,
            ChallengeMmcs,
            <ValMmcs as Mmcs<Val>>::Error,
        >(
            &folding,
            &pcs.fri.mmcs,
            &mut start_index,
            fold_data_iter,
            reduced_openings,
            protocol.log_max_height,
            log_final_height(&pcs.fri),
        )
        .expect_err("missing initial reduced opening should fail");

        match err {
            FriError::MissingInitialReducedOpening { .. } => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn initial_reduced_opening_height_mismatch_hits_fri_error() {
        let (pcs, protocol, _ds, _proof) = minimal_proof();
        let folding = TwoAdicFriFolding;
        let wrong_height = protocol.log_max_height - 1;
        let reduced_openings: FriOpenings<Challenge> =
            vec![(wrong_height, Challenge::from(Val::from_u8(7)))];
        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<crate::CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data_iter = betas.iter().zip(commits.iter()).zip(openings.iter());
        let mut start_index = 0;

        let err: TestError = verify_query::<
            TwoAdicFriFolding,
            Val,
            Challenge,
            ChallengeMmcs,
            <ValMmcs as Mmcs<Val>>::Error,
        >(
            &folding,
            &pcs.fri.mmcs,
            &mut start_index,
            fold_data_iter,
            reduced_openings,
            protocol.log_max_height,
            log_final_height(&pcs.fri),
        )
        .expect_err("wrong initial reduced opening height should fail");

        match err {
            FriError::InitialReducedOpeningHeightMismatch { got, .. } => {
                assert_eq!(got, wrong_height);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn final_fold_height_mismatch_hits_fri_error() {
        let (pcs, protocol, _ds, _proof) = minimal_proof();
        let folding = TwoAdicFriFolding;
        let reduced_openings: FriOpenings<Challenge> =
            vec![(protocol.log_max_height, Challenge::from(Val::from_u8(42)))];
        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<crate::CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data_iter = betas.iter().zip(commits.iter()).zip(openings.iter());
        let mut start_index = 0;

        let err: TestError = verify_query::<
            TwoAdicFriFolding,
            Val,
            Challenge,
            ChallengeMmcs,
            <ValMmcs as Mmcs<Val>>::Error,
        >(
            &folding,
            &pcs.fri.mmcs,
            &mut start_index,
            fold_data_iter,
            reduced_openings,
            protocol.log_max_height,
            log_final_height(&pcs.fri),
        )
        .expect_err("unfolded chain should not reach final height");

        match err {
            FriError::FinalFoldHeightMismatch { .. } => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn unconsumed_reduced_openings_hits_fri_error() {
        let (pcs, protocol, _ds, _proof) = minimal_proof();
        let folding = TwoAdicFriFolding;
        let reduced_openings: FriOpenings<Challenge> = vec![
            (protocol.log_max_height, Challenge::from(Val::from_u8(42))),
            (
                protocol.log_max_height - 1,
                Challenge::from(Val::from_u8(99)),
            ),
        ];
        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<crate::CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data_iter = betas.iter().zip(commits.iter()).zip(openings.iter());
        let mut start_index = 0;

        let err: TestError = verify_query::<
            TwoAdicFriFolding,
            Val,
            Challenge,
            ChallengeMmcs,
            <ValMmcs as Mmcs<Val>>::Error,
        >(
            &folding,
            &pcs.fri.mmcs,
            &mut start_index,
            fold_data_iter,
            reduced_openings,
            protocol.log_max_height,
            protocol.log_max_height,
        )
        .expect_err("leftover reduced opening should fail");

        match err {
            FriError::UnconsumedReducedOpenings { .. } => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    fn tamper_field_at_label(ds: &DomainSeparator, label: &'static str, proof: &Proof) -> Proof {
        let mut proof = proof.clone();
        let offset = ds
            .byte_offset_for_label::<Val, Challenge, FieldToFieldCodec<Val>, Challenger>(label)
            .expect("label should carry proof bytes");
        proof[offset + field_byte_size::<Val>() - 1] ^= 1;
        proof
    }

    fn tamper_pow_at_label(ds: &DomainSeparator, label: &'static str, proof: &Proof) -> Proof {
        let mut proof = proof.clone();
        let offset = ds
            .byte_offset_for_label::<Val, Challenge, FieldToFieldCodec<Val>, Challenger>(label)
            .expect("PoW label should carry proof bytes");
        assert!(offset + field_byte_size::<Val>() <= proof.len());
        proof[offset] |= 0x80;
        proof
    }
}

mod hiding {
    use super::*;

    const HIDING_SALT_ELEMS: usize = 4;
    type HidingValMmcs = MerkleTreeHidingMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        SmallRng,
        2,
        8,
        HIDING_SALT_ELEMS,
    >;
    type HidingChallengeMmcs = ExtensionMmcs<Val, Challenge, HidingValMmcs>;
    type HidingPcs =
        HidingFriPcs<Val, Radix2Dit<Val>, HidingValMmcs, HidingChallengeMmcs, SmallRng>;
    type Domain = <HidingPcs as Pcs<Challenge, Challenger>>::Domain;
    type HidingTestError = FriError<
        <HidingChallengeMmcs as p3_commit::Mmcs<Challenge>>::Error,
        <HidingValMmcs as p3_commit::Mmcs<Val>>::Error,
    >;

    fn hiding_fri_mmcs(rng: SmallRng) -> HidingChallengeMmcs {
        let perm = perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        HidingChallengeMmcs::new(HidingValMmcs::new(hash.clone(), compress.clone(), 0, rng))
    }

    fn hiding_input_mmcs(rng: SmallRng) -> HidingValMmcs {
        let perm = perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        HidingValMmcs::new(hash.clone(), compress.clone(), 0, rng)
    }

    fn rand_hiding_matrix<R: RngExt>(
        rng: &mut R,
        spec: &BatchSpec,
        pcs: &HidingPcs,
    ) -> Vec<(Domain, RowMajorMatrix<Val>)> {
        spec.matrices()
            .iter()
            .map(|matrix_spec| {
                let dimensions = matrix_spec.dimensions();
                (
                    <HidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                        pcs,
                        dimensions.height << 1,
                    ),
                    RowMajorMatrix::<Val>::rand_nonzero(rng, dimensions.height, dimensions.width),
                )
            })
            .collect::<Vec<_>>()
    }

    fn run_hiding_prover<R: RngExt>(
        rng: &mut R,
        pcs: &HidingPcs,
        protocol: &FriProtocol,
        transcript: &mut ProverState<Challenger>,
    ) {
        let matrices = protocol
            .batches
            .iter()
            .map(|batch| rand_hiding_matrix(rng, batch, pcs))
            .collect::<Vec<_>>();

        let (commitments, data) = matrices
            .into_iter()
            .map(|mat| <HidingPcs as Pcs<Challenge, Challenger>>::commit(pcs, mat))
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let mut opening_points = Vec::with_capacity(protocol.batches.len());
        for (batch_index, (commitment, batch)) in
            commitments.into_iter().zip(&protocol.batches).enumerate()
        {
            pcs.input_mmcs().write_commitment(
                transcript,
                Labels::commitment(batch_index),
                commitment,
            );

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

        pcs.open(
            data.iter().zip(opening_points).collect::<Vec<_>>(),
            transcript,
        );
    }

    fn run_hiding_verifier(
        pcs: &HidingPcs,
        protocol: &FriProtocol,
        transcript: &mut VerifierState<'_, Challenger>,
    ) -> Result<(), HidingTestError> {
        let mut commitments = Vec::with_capacity(protocol.batches.len());
        let mut opening_points = Vec::with_capacity(protocol.batches.len());
        for (batch_index, batch) in protocol.batches.iter().enumerate() {
            let commitment = pcs
                .input_mmcs()
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
                        let domain =
                            <HidingPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                                pcs,
                                matrix.height() << 1,
                            );
                        let points_and_values = opening_points[batch_index][matrix_index]
                            .iter()
                            .enumerate()
                            .map(|(point_index, &point)| {
                                let values = transcript
                                    .next_extensions::<Val, Challenge, FieldToFieldCodec<Val>>(
                                        Labels::opened_values(
                                            batch_index,
                                            matrix_index,
                                            point_index,
                                        ),
                                        matrix.width() + pcs.num_random_codewords(),
                                    )?
                                    .into_iter()
                                    .map(|value| value.into_inner())
                                    .collect::<Vec<_>>();
                                Ok((point, values))
                            })
                            .collect::<Result<Vec<_>, HidingTestError>>()?;
                        Ok((domain, points_and_values))
                    })
                    .collect::<Result<Vec<_>, HidingTestError>>()?;
                Ok((commitment, batch_opening_points))
            })
            .collect::<Result<Vec<_>, HidingTestError>>()?;

        pcs.verify(commitments_with_opening_points, transcript)
    }

    fn hiding_prover<R: RngExt>(rng: &mut R, pcs: &HidingPcs, protocol: &FriProtocol) -> Proof {
        let ds = domain_separator(&protocol, pcs.params(), pcs.input_mmcs());
        let mut transcript = ProverState::<_, u8>::new(challenger(), &ds);
        run_hiding_prover(rng, pcs, &protocol, &mut transcript);
        transcript.finalize()
    }

    fn hiding_verifier(
        pcs: &HidingPcs,
        protocol: &FriProtocol,
        proof: Proof,
    ) -> Result<(), HidingTestError> {
        let ds = domain_separator(&protocol, pcs.params(), pcs.input_mmcs());
        let mut transcript = VerifierState::<_, u8>::new(challenger(), &ds, &proof);
        run_hiding_verifier(pcs, &protocol, &mut transcript)?;
        transcript.finalize().map_err(FriError::Transcript)
    }

    fn run_hiding_rand_test(params: FriParameters<HidingChallengeMmcs>) {
        let num_random_codewords = 2;
        let mut rng = SmallRng::seed_from_u64(11);
        let input_mmcs = hiding_input_mmcs(SmallRng::seed_from_u64(17));
        let pcs = HidingPcs::new(
            Radix2Dit::default(),
            input_mmcs,
            params,
            num_random_codewords,
            SmallRng::seed_from_u64(19),
        );

        for _ in 0..100 {
            let batches = random_batches(&mut rng);
            let protocol = FriProtocol::new_hiding(
                pcs.params(),
                batches,
                Folding::extra_query_index_bits(),
                pcs.num_random_codewords(),
            );
            let proof = hiding_prover(&mut rng, &pcs, &protocol);
            hiding_verifier(&pcs, &protocol, proof).unwrap();
        }
    }

    #[test]
    fn test_hiding_rand_fixtures() {
        let params = FriParameters {
            mmcs: hiding_fri_mmcs(SmallRng::seed_from_u64(13)),
            log_blowup: 2,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 5,
            query_proof_of_work_bits: 5,
        };
        run_hiding_rand_test(params);
    }

    #[test]
    fn test_hiding_minimal_fixture() {
        let num_random_codewords = 2;
        let params = FriParameters::new_testing(hiding_fri_mmcs(SmallRng::seed_from_u64(13)), 0);
        let mut rng = SmallRng::seed_from_u64(11);
        let input_mmcs = hiding_input_mmcs(SmallRng::seed_from_u64(17));
        let pcs = HidingPcs::new(
            Radix2Dit::default(),
            input_mmcs,
            params,
            num_random_codewords,
            SmallRng::seed_from_u64(19),
        );

        let batches = vec![minimal_batch()];
        let protocol = FriProtocol::new_hiding(
            pcs.params(),
            batches,
            Folding::extra_query_index_bits(),
            pcs.num_random_codewords(),
        );
        let proof = hiding_prover(&mut rng, &pcs, &protocol);
        hiding_verifier(&pcs, &protocol, proof).unwrap();
    }
}
