use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, InteractionPattern, ProverState, VerifierState,
};
use p3_commit::{ExtensionMmcs, MmcsReader, MmcsWriter, Pcs};
use p3_dft::Radix2Dit;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::protocol::{
    BatchSpec, FriLabels, FriLabelsDefault as Labels, FriProtocol, MatrixSpec, Protocol as _,
};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicFriPcs<BabyBear, Radix2Dit<BabyBear>, ValMmcs, ChallengeMmcs>;
type Domain = <MyPcs as Pcs<Challenge, Challenger>>::Domain;

const WIDTDH: usize = 16;

fn domain_separator(
    protocol: &FriProtocol,
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
    let mut ds: DomainSeparator<u8> = DomainSeparator::new(
        1,
        b"fri-test",
        InteractionPattern::new(interactions).unwrap(),
    );
    ds.bind_pattern_hash();
    ds
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

fn input_mmcs() -> ValMmcs {
    let perm = perm();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    ValMmcs::new(hash.clone(), compress.clone(), 0)
}

fn fri_mmcs() -> ChallengeMmcs {
    let perm = perm();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    ChallengeMmcs::new(ValMmcs::new(hash.clone(), compress.clone(), 0))
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

/// Check that the loop of `pcs.commit`, `pcs.open`, and `pcs.verify` work correctly.
///
/// We create a random polynomial of size `1 << log_size` for each size in `polynomial_log_sizes`.
/// We then commit to these polynomials using a `log_blowup` of `1`.
///
/// We open each polynomial at the same point `zeta` and run FRI to verify the openings, stopping
/// FRI at `log_final_poly_len`.
fn do_test_fri_ldt<R: Rng>(rng: &mut R, log_final_poly_len: usize, polynomial_log_sizes: &[u8]) {
    let input_mmcs = input_mmcs();
    let fri_mmcs = fri_mmcs();
    let params = FriParameters {
        mmcs: fri_mmcs,
        log_blowup: 1,
        log_final_poly_len,
        max_log_arity: 1,
        num_queries: 10,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
    };

    let matrices = polynomial_log_sizes
        .iter()
        .map(|&log_size| {
            MatrixSpec::new(
                Dimensions {
                    height: 1 << log_size,
                    width: WIDTDH,
                },
                1,
            )
        })
        .collect::<Vec<_>>();
    let protocol = FriProtocol::new(&params, vec![BatchSpec::new(matrices)], 0);
    let pcs = MyPcs::new(Radix2Dit::default(), input_mmcs, params);
    let ds = domain_separator(&protocol, pcs.params(), pcs.input_mmcs());

    // --- Prover World ---
    let proof = {
        let mut transcript = ProverState::<_, u8>::new(challenger(), &ds);
        let matrices = protocol
            .batches
            .iter()
            .map(|batch| rand_matrix(rng, batch, &pcs))
            .collect::<Vec<_>>();
        let (commitments, data) = matrices
            .into_iter()
            .map(|mat| <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, mat))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let mut opening_points = Vec::with_capacity(protocol.batches.len());
        for (batch_index, (commitment, batch)) in
            commitments.into_iter().zip(&protocol.batches).enumerate()
        {
            pcs.input_mmcs().write_commitment(
                &mut transcript,
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
            &mut transcript,
        );

        transcript.finalize()
    };

    // --- Verifier World ---
    {
        let mut transcript = VerifierState::<_, u8>::new(challenger(), &ds, &proof);
        let mut commitments = Vec::with_capacity(protocol.batches.len());
        let mut opening_points = Vec::with_capacity(protocol.batches.len());
        for (batch_index, batch) in protocol.batches.iter().enumerate() {
            let commitment = pcs
                .input_mmcs()
                .read_commitment(
                    &mut transcript,
                    Labels::commitment(batch_index),
                    &protocol.input_batch_dimensions[batch_index],
                )
                .unwrap()
                .into_inner();
            commitments.push(commitment);

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

        let commitments_with_opening_points = protocol
            .batches
            .iter()
            .zip(commitments)
            .enumerate()
            .map(|(batch_index, (batch, commitment))| {
                let batch_opening_points = batch
                    .matrices()
                    .iter()
                    .enumerate()
                    .map(|(matrix_index, matrix)| {
                        let domain =
                            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                                &pcs,
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
                                    )
                                    .unwrap()
                                    .into_iter()
                                    .map(|value| value.into_inner())
                                    .collect::<Vec<_>>();
                                (point, values)
                            })
                            .collect::<Vec<_>>();
                        (domain, points_and_values)
                    })
                    .collect::<Vec<_>>();
                (commitment, batch_opening_points)
            })
            .collect::<Vec<_>>();

        pcs.verify(commitments_with_opening_points, &mut transcript)
            .unwrap();
        transcript.finalize().unwrap();
    }
}

/// Test that the FRI commit, open and verify process work correctly
/// for a range of `final_poly_degree` values.
#[test]
fn test_fri_ldt() {
    // Chosen to ensure there are both multiple polynomials
    // of the same size and that the array is not ordered.
    let polynomial_log_sizes = [5, 8, 10, 7, 5, 5, 7];
    for i in 0..5 {
        let mut rng = SmallRng::seed_from_u64(i as u64);
        do_test_fri_ldt(&mut rng, i, &polynomial_log_sizes);
    }
}

/// This test is expected to panic because there is a polynomial degree which
/// the prover commits too which is less than `final_poly_degree`.
#[test]
#[should_panic]
fn test_fri_ldt_should_panic() {
    // Chosen to ensure there are both multiple polynomials
    // of the same size and that the array is not ordered.
    let polynomial_log_sizes = [5, 8, 10, 7, 5, 5, 7];
    let mut rng = SmallRng::seed_from_u64(5);
    do_test_fri_ldt(&mut rng, 5, &polynomial_log_sizes);
}
