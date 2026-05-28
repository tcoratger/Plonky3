use itertools::Itertools;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::fs::{
    CanObserveBytes, DefaultCodec, DomainSeparator, FieldToFieldCodec, InteractionPattern,
    ProverState, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, DuplexChallenger};
use p3_commit::{ExtensionMmcs, MmcsReader, MmcsTranscript, MmcsWriter, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField};
use p3_fri::protocol::{BatchSpec, FriLabels, MatrixSpec, Protocol};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

fn seeded_rng() -> impl Rng {
    SmallRng::seed_from_u64(0)
}

fn domain_separator<Val, Challenge, InputMmcs, FriMmcs, P>(
    protocol: &P,
    params: &FriParameters<FriMmcs>,
    input_mmcs: &InputMmcs,
) -> DomainSeparator<u8>
where
    Val: PrimeCharacteristicRing + Send + Sync + Clone,
    Challenge: BasedVectorSpace<Val> + Send + Sync + Clone,
    InputMmcs: MmcsTranscript<Val>,
    FriMmcs: MmcsTranscript<Challenge>,
    P: Protocol,
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
    let mut ds: DomainSeparator<u8> =
        DomainSeparator::new(1, b"test", InteractionPattern::new(interactions).unwrap());
    ds.bind_pattern_hash();
    ds
}

fn do_test_pcs<Val, Challenge, Challenger, P, InputMmcs, FriMmcs, ProtocolImpl, LabelsImpl>(
    (pcs, challenger, params, input_mmcs, extra_query_index_bits): (
        &P,
        &Challenger,
        &FriParameters<FriMmcs>,
        &InputMmcs,
        usize,
    ),
    log_degrees_by_round: &[&[usize]],
) where
    P: Pcs<Challenge, Challenger, Commitment = InputMmcs::Commitment>,
    P::Domain: PolynomialSpace<Val = Val>,
    Val: Field + PrimeField + PrimeCharacteristicRing + Send + Sync + Clone,
    StandardUniform: Distribution<Val>,
    Challenge: ExtensionField<Val> + BasedVectorSpace<Val> + Send + Sync + Clone,
    Challenger: Clone
        + CanObserveBytes
        + CanObserve<Val>
        + CanSample<Val>
        + DefaultCodec<InputMmcs::Digest>
        + DefaultCodec<FriMmcs::Digest>,
    InputMmcs: MmcsWriter<Val> + MmcsReader<Val>,
    FriMmcs: MmcsTranscript<Challenge>,
    ProtocolImpl: Protocol,
    LabelsImpl: FriLabels,
{
    let num_rounds = log_degrees_by_round.len();
    let mut rng = seeded_rng();

    let batches = log_degrees_by_round
        .iter()
        .map(|log_degrees| {
            BatchSpec::new(
                log_degrees
                    .iter()
                    .map(|&log_degree| {
                        let height = 1 << log_degree;
                        // random width 5-15
                        let width = 5 + rng.random_range(0..=10);
                        MatrixSpec::new(Dimensions { width, height }, 1)
                    })
                    .collect_vec(),
            )
        })
        .collect_vec();
    let protocol = ProtocolImpl::new(params, batches, extra_query_index_bits, None);
    let ds =
        domain_separator::<Val, Challenge, InputMmcs, FriMmcs, _>(&protocol, params, input_mmcs);

    let domains_and_polys_by_round = protocol
        .batches()
        .iter()
        .map(|batch| {
            batch
                .matrices()
                .iter()
                .map(|matrix| {
                    let dimensions = matrix.dimensions();
                    (
                        pcs.natural_domain_for_degree(dimensions.height),
                        RowMajorMatrix::<Val>::rand(&mut rng, dimensions.height, dimensions.width),
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    // Prover transcript.
    let proof = {
        let mut transcript = ProverState::<_, u8>::new(Challenger::clone(challenger), &ds);
        let (commits_by_round, data_by_round): (Vec<_>, Vec<_>) = domains_and_polys_by_round
            .iter()
            .map(|domains_and_polys| pcs.commit(domains_and_polys.iter().cloned()))
            .unzip();
        assert_eq!(commits_by_round.len(), num_rounds);
        assert_eq!(data_by_round.len(), num_rounds);

        let mut opening_points = Vec::with_capacity(protocol.batches().len());
        for (batch_index, (commitment, batch)) in commits_by_round
            .into_iter()
            .zip(protocol.batches())
            .enumerate()
        {
            input_mmcs.write_commitment(
                &mut transcript,
                LabelsImpl::commitment(batch_index),
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
                                    LabelsImpl::opening_point(
                                        batch_index,
                                        matrix_index,
                                        point_index,
                                    ),
                                )
                                .into_inner()
                        })
                        .collect_vec()
                })
                .collect_vec();
            opening_points.push(batch_opening_points);
        }

        pcs.open(
            data_by_round.iter().zip(opening_points).collect_vec(),
            &mut transcript,
        );
        transcript.finalize()
    };

    // Verifier transcript.
    {
        let mut transcript =
            VerifierState::<_, u8>::new(Challenger::clone(challenger), &ds, &proof);
        let mut commits_by_round = Vec::with_capacity(protocol.batches().len());
        let mut opening_points = Vec::with_capacity(protocol.batches().len());
        for (batch_index, batch) in protocol.batches().iter().enumerate() {
            let commitment = input_mmcs
                .read_commitment(
                    &mut transcript,
                    LabelsImpl::commitment(batch_index),
                    &protocol.input_batch_dimensions()[batch_index],
                )
                .unwrap()
                .into_inner();
            commits_by_round.push(commitment);

            let batch_opening_points = batch
                .matrices()
                .iter()
                .enumerate()
                .map(|(matrix_index, matrix)| {
                    (0..matrix.num_opening_points())
                        .map(|point_index| {
                            transcript
                                .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                                    LabelsImpl::opening_point(
                                        batch_index,
                                        matrix_index,
                                        point_index,
                                    ),
                                )
                                .into_inner()
                        })
                        .collect_vec()
                })
                .collect_vec();
            opening_points.push(batch_opening_points);
        }

        let commits_and_claims_by_round = protocol
            .batches()
            .iter()
            .zip(commits_by_round)
            .enumerate()
            .map(|(batch_index, (batch, commitment))| {
                let claims = batch
                    .matrices()
                    .iter()
                    .enumerate()
                    .map(|(matrix_index, matrix)| {
                        let domain = pcs.natural_domain_for_degree(matrix.height());
                        let points_and_values = opening_points[batch_index][matrix_index]
                            .iter()
                            .enumerate()
                            .map(|(point_index, &point)| {
                                let values = transcript
                                    .next_extensions::<Val, Challenge, FieldToFieldCodec<Val>>(
                                        LabelsImpl::opened_values(
                                            batch_index,
                                            matrix_index,
                                            point_index,
                                        ),
                                        matrix.width(),
                                    )
                                    .unwrap()
                                    .into_iter()
                                    .map(|value| value.into_inner())
                                    .collect_vec();
                                (point, values)
                            })
                            .collect_vec();
                        (domain, points_and_values)
                    })
                    .collect_vec();
                (commitment, claims)
            })
            .collect_vec();
        assert_eq!(commits_and_claims_by_round.len(), num_rounds);

        pcs.verify(commits_and_claims_by_round, &mut transcript)
            .unwrap();
        transcript.finalize().unwrap();
    }
}

// Set it up so we create tests inside a module for each pcs, so we get nice error reports
// specific to a failing PCS.
macro_rules! make_tests_for_pcs {
    ($p:expr, $protocol:ty, $labels:ty, $extra_query_index_bits:expr) => {
        #[test]
        fn single() {
            let (pcs, challenger, params, input_mmcs) = $p;
            for i in 3..6 {
                let test_params = (
                    &pcs,
                    &challenger,
                    &params,
                    &input_mmcs,
                    $extra_query_index_bits,
                );
                $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(test_params, &[&[i]]);
            }
        }

        #[test]
        fn many_equal() {
            let (pcs, challenger, params, input_mmcs) = $p;
            for i in 2..6 {
                let test_params = (
                    &pcs,
                    &challenger,
                    &params,
                    &input_mmcs,
                    $extra_query_index_bits,
                );
                $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                    test_params,
                    &[&[i; 5]],
                );
                println!("{i} ok");
            }
        }

        #[test]
        fn many_different() {
            let (pcs, challenger, params, input_mmcs) = $p;
            for i in 2..5 {
                let degrees = (3..3 + i).collect::<Vec<_>>();
                let test_params = (
                    &pcs,
                    &challenger,
                    &params,
                    &input_mmcs,
                    $extra_query_index_bits,
                );
                $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                    test_params,
                    &[&degrees],
                );
            }
        }

        #[test]
        fn many_different_rev() {
            let (pcs, challenger, params, input_mmcs) = $p;
            for i in 2..5 {
                let degrees = (3..3 + i).rev().collect::<Vec<_>>();
                let test_params = (
                    &pcs,
                    &challenger,
                    &params,
                    &input_mmcs,
                    $extra_query_index_bits,
                );
                $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                    test_params,
                    &[&degrees],
                );
            }
        }

        #[test]
        fn multiple_rounds() {
            let (pcs, challenger, params, input_mmcs) = $p;
            let test_params = || {
                (
                    &pcs,
                    &challenger,
                    &params,
                    &input_mmcs,
                    $extra_query_index_bits,
                )
            };
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(test_params(), &[&[3]]);
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                test_params(),
                &[&[3], &[3]],
            );
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                test_params(),
                &[&[3], &[2]],
            );
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                test_params(),
                &[&[2], &[3]],
            );
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                test_params(),
                &[&[3, 4], &[3, 4]],
            );
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                test_params(),
                &[&[4, 2], &[4, 2]],
            );
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                test_params(),
                &[&[2, 2], &[3, 3]],
            );
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                test_params(),
                &[&[3, 3], &[2, 2]],
            );
            $crate::do_test_pcs::<_, _, _, _, _, _, $protocol, $labels>(
                test_params(),
                &[&[2], &[3, 3]],
            );
        }
    };
}

mod babybear_fri_pcs {
    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    fn get_pcs(log_blowup: usize) -> (MyPcs, Challenger, FriParameters<ChallengeMmcs>, ValMmcs) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        let fri_params = FriParameters {
            mmcs: challenge_mmcs,
            log_blowup,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 10,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 8,
        };

        let pcs = MyPcs::new(Dft::default(), val_mmcs.clone(), fri_params.clone());
        (pcs, Challenger::new(perm), fri_params, val_mmcs)
    }

    fn get_pcs_high_arity(
        log_blowup: usize,
    ) -> (MyPcs, Challenger, FriParameters<ChallengeMmcs>, ValMmcs) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        let fri_params = FriParameters {
            mmcs: challenge_mmcs,
            log_blowup,
            log_final_poly_len: 0,
            max_log_arity: 2,
            num_queries: 10,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 8,
        };

        let pcs = MyPcs::new(Dft::default(), val_mmcs.clone(), fri_params.clone());
        (pcs, Challenger::new(perm), fri_params, val_mmcs)
    }

    mod blowup_1 {
        make_tests_for_pcs!(
            super::get_pcs(1),
            p3_fri::protocol::FriProtocol,
            p3_fri::protocol::FriLabelsDefault,
            0
        );
    }
    mod blowup_2 {
        make_tests_for_pcs!(
            super::get_pcs(2),
            p3_fri::protocol::FriProtocol,
            p3_fri::protocol::FriLabelsDefault,
            0
        );
    }
    mod high_arity_blowup_1 {
        make_tests_for_pcs!(
            super::get_pcs_high_arity(1),
            p3_fri::protocol::FriProtocol,
            p3_fri::protocol::FriLabelsDefault,
            0
        );
    }

    #[test]
    fn extrapolation() {
        use p3_dft::TwoAdicSubgroupDft;
        use p3_matrix::Matrix;

        let (pcs, _, _, _) = get_pcs(1);
        let mut rng = seeded_rng();

        let log_degree = 4;
        let degree = 1 << log_degree;
        let width = 3;
        let trace = RowMajorMatrix::<Val>::rand(&mut rng, degree, width);

        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree);
        let (_, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain, trace.clone())]);

        let disjoint_domain = domain.create_disjoint_domain(degree);
        let evals = <MyPcs as Pcs<Challenge, Challenger>>::get_evaluations_on_domain(
            &pcs,
            &data,
            0,
            disjoint_domain,
        );
        let evals = evals.to_row_major_matrix();

        let dft = Dft::default();
        let coeffs = dft.idft_batch(trace);
        let expected = dft
            .coset_dft_batch(coeffs, disjoint_domain.shift())
            .to_row_major_matrix();

        assert_eq!(evals, expected);
    }
}

mod babybear_fri_pcs_keccak {
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_keccak::Keccak256Hash;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<BabyBear, 4>;
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    type Dft = Radix2DitParallel<Val>;
    type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    fn get_pcs(log_blowup: usize) -> (MyPcs, Challenger, FriParameters<ChallengeMmcs>, ValMmcs) {
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        let fri_params = FriParameters {
            mmcs: challenge_mmcs,
            log_blowup,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 10,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 8,
        };

        let pcs = MyPcs::new(Dft::default(), val_mmcs.clone(), fri_params.clone());
        (
            pcs,
            Challenger::from_hasher(vec![], byte_hash),
            fri_params,
            val_mmcs,
        )
    }

    mod blowup_1 {
        make_tests_for_pcs!(
            super::get_pcs(1),
            p3_fri::protocol::FriProtocol,
            p3_fri::protocol::FriLabelsDefault,
            0
        );
    }
}

mod m31_fri_pcs {
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_circle::CirclePcs;
    use p3_keccak::Keccak256Hash;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

    use super::*;

    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Mersenne31, 3>;
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    type MyPcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;

    const EXTRA_QUERY_INDEX_BITS: usize = 1;

    fn get_pcs(log_blowup: usize) -> (MyPcs, Challenger, FriParameters<ChallengeMmcs>, ValMmcs) {
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);
        let compress = MyCompress::new(byte_hash);
        let val_mmcs = ValMmcs::new(field_hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        let fri_params = FriParameters {
            mmcs: challenge_mmcs,
            log_blowup,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 10,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 8,
        };

        let pcs = MyPcs::new(val_mmcs.clone(), fri_params.clone());
        (
            pcs,
            Challenger::from_hasher(vec![], byte_hash),
            fri_params,
            val_mmcs,
        )
    }

    mod blowup_1 {
        make_tests_for_pcs!(
            super::get_pcs(1),
            p3_circle::protocol::CircleProtocol,
            p3_circle::protocol::CircleLabelsDefault,
            super::EXTRA_QUERY_INDEX_BITS
        );
    }
    mod blowup_2 {
        make_tests_for_pcs!(
            super::get_pcs(2),
            p3_circle::protocol::CircleProtocol,
            p3_circle::protocol::CircleLabelsDefault,
            super::EXTRA_QUERY_INDEX_BITS
        );
    }
}
