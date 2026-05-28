use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_challenger::fs::{
    DefaultCodec, ExtensionFieldCodec, FieldToFieldCodec, ProverState, TranscriptError,
    VerifierState,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{
    BatchDimensions, BatchOpening, BatchOpeningRef, BuildPeriodicLdeTableFast, Mmcs, MmcsReader,
    MmcsWriter, OpenedValues, Pcs, PeriodicLdeTable, PolynomialSpace,
};
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field, PrimeField};
use p3_fri::verifier::FriError;
use p3_fri::{CommitmentWithOpeningPoints, FriParameters};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::row_index_mapped::RowIndexMappedView;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use p3_util::zip_eq::zip_eq;
use thiserror::Error;
use tracing::info_span;

use crate::deep_quotient::{deep_quotient_reduce_row, extract_lambda};
use crate::domain::CircleDomain;
use crate::folding::{CircleFriFolding, fold_y, fold_y_row};
use crate::point::Point;
use crate::protocol::{BatchSpec, CircleLabels, CircleLabelsDefault, CircleProtocol, FriLabels};
use crate::prover::prove;
use crate::verifier::verify;
use crate::{
    CfftPerm, CfftPermutable, CircleEvaluations, build_periodic_lde_table_circle,
    cfft_permute_index,
};

#[derive(Clone, Debug)]
pub struct CirclePcs<Val: Field, InputMmcs, FriMmcs> {
    pub mmcs: InputMmcs,
    pub fri_params: FriParameters<FriMmcs>,
    pub _phantom: PhantomData<Val>,
}

impl<Val: Field, InputMmcs, FriMmcs> CirclePcs<Val, InputMmcs, FriMmcs> {
    pub const fn new(mmcs: InputMmcs, fri_params: FriParameters<FriMmcs>) -> Self {
        Self {
            mmcs,
            fri_params,
            _phantom: PhantomData,
        }
    }
}

#[derive(Debug, Error)]
pub enum InputError<InputMmcsError, FriMmcsError>
where
    InputMmcsError: core::fmt::Debug,
    FriMmcsError: core::fmt::Debug,
{
    #[error("input MMCS error: {0:?}")]
    InputMmcsError(InputMmcsError),
    #[error("first layer MMCS error: {0:?}")]
    FirstLayerMmcsError(FriMmcsError),
    #[error("input shape error: mismatched dimensions")]
    InputShapeError,
}

impl<Val, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for CirclePcs<Val, InputMmcs, FriMmcs>
where
    Val: ComplexExtendable + PrimeField,
    Challenge: ExtensionField<Val>,
    InputMmcs: MmcsWriter<Val> + MmcsReader<Val, Proof: Sync, Error: Sync>,
    FriMmcs: MmcsWriter<Challenge> + MmcsReader<Challenge>,
    Challenger: FieldChallenger<Val>
        + GrindingChallenger<Witness = Val>
        + DefaultCodec<InputMmcs::Digest>
        + DefaultCodec<FriMmcs::Digest>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = RowIndexMappedView<CfftPerm, RowMajorMatrixCow<'a, Val>>;
    type Error = FriError<FriMmcs::Error, InputError<InputMmcs::Error, FriMmcs::Error>>;
    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        CircleDomain::standard(log2_strict_usize(degree))
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let ldes = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert!(
                    domain.log_n >= 2,
                    "CirclePcs cannot commit to a matrix with fewer than 4 rows.",
                    // (because we bivariate fold one bit, and fri needs one more bit)
                );
                CircleEvaluations::from_natural_order(domain, evals)
                    .extrapolate(CircleDomain::standard(
                        domain.log_n + self.fri_params.log_blowup,
                    ))
                    .to_cfft_order()
            })
            .collect_vec();
        let (comm, mmcs_data) = self.mmcs.commit(ldes);
        (comm, mmcs_data)
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert!(
                    domain.log_n >= 2,
                    "CirclePcs cannot commit to a matrix with fewer than 4 rows.",
                    // (because we bivariate fold one bit, and fri needs one more bit)
                );
                CircleEvaluations::from_natural_order(domain, evals)
                    .extrapolate(CircleDomain::standard(
                        domain.log_n + self.fri_params.log_blowup,
                    ))
                    .to_cfft_order()
            })
            .collect_vec()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        self.mmcs.commit(ldes)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let mat = self.mmcs.get_matrices(data)[idx].as_view();
        let committed_domain = CircleDomain::standard(log2_strict_usize(mat.height()));
        if domain == committed_domain {
            mat.as_cow().cfft_perm_rows()
        } else {
            CircleEvaluations::from_cfft_order(committed_domain, mat)
                .extrapolate(domain)
                .to_cfft_order()
                .as_cow()
                .cfft_perm_rows()
        }
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        transcript: &mut ProverState<Challenger>,
    ) {
        // Open matrices at points
        let values: OpenedValues<Challenge> = rounds
            .iter()
            .enumerate()
            .map(|(batch_index, (data, points_for_mats))| {
                let mats = self.mmcs.get_matrices(data);
                debug_assert_eq!(
                    mats.len(),
                    points_for_mats.len(),
                    "Mismatched number of matrices and points"
                );
                izip!(mats, points_for_mats)
                    .enumerate()
                    .map(|(matrix_index, (mat, points_for_mat))| {
                        let log_height = log2_strict_usize(mat.height());
                        // It was committed in cfft order.
                        let evals = CircleEvaluations::from_cfft_order(
                            CircleDomain::standard(log_height),
                            mat.as_view(),
                        );
                        points_for_mat
                            .iter()
                            .enumerate()
                            .map(|(point_index, &zeta)| {
                                let zeta = Point::from_projective_line(zeta);
                                let ps_at_zeta =
                                    info_span!("compute opened values with Lagrange interpolation")
                                        .in_scope(|| evals.evaluate_at_point(zeta));
                                transcript
                                    .add_extensions::<Val, Challenge, FieldToFieldCodec<Val>>(
                                        CircleLabelsDefault::opened_values(
                                            batch_index,
                                            matrix_index,
                                            point_index,
                                        ),
                                        &ps_at_zeta,
                                    );
                                ps_at_zeta
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // Batch combination challenge
        let alpha = transcript
            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                CircleLabelsDefault::alpha(),
            )
            .into_inner();

        /*
        We are reducing columns ("ro" = reduced opening) with powers of alpha:
          ro = .. + α^n c_n + α^(n+1) c_(n+1) + ..
        But we want to precompute small powers of alpha, and batch the columns. So we can do:
          ro = .. + α^n (α^0 c_n + α^1 c_(n+1) + ..) + ..
        reusing the α^0, α^1, etc., then at the end of each column batch we multiply by the α^n.
        (Due to circle stark specifics, we need 2 powers of α for each column, so actually α^(2n)).
        We store this α^(2n), the running reducing factor per log_height, and call it the "alpha offset".
        */

        // log_height -> (alpha offset, reduced openings column)
        let mut reduced_openings: BTreeMap<usize, (Challenge, Vec<Challenge>)> = BTreeMap::new();

        rounds
            .iter()
            .zip(values.iter())
            .for_each(|((data, points_for_mats), values)| {
                let mats = self.mmcs.get_matrices(data);
                izip!(mats, points_for_mats, values).for_each(|(mat, points_for_mat, values)| {
                    let log_height = log2_strict_usize(mat.height());
                    // It was committed in cfft order.
                    let evals = CircleEvaluations::from_cfft_order(
                        CircleDomain::standard(log_height),
                        mat.as_view(),
                    );

                    let (alpha_offset, reduced_opening_for_log_height) =
                        reduced_openings.entry(log_height).or_insert_with(|| {
                            (Challenge::ONE, vec![Challenge::ZERO; 1 << log_height])
                        });

                    points_for_mat
                        .iter()
                        .zip(values.iter())
                        .for_each(|(&zeta, ps_at_zeta)| {
                            let zeta = Point::from_projective_line(zeta);

                            // Reduce this matrix, as a deep quotient, into one column with powers of α.
                            let mat_ros = evals.deep_quotient_reduce(alpha, zeta, ps_at_zeta);

                            // Fold it into our running reduction, offset by alpha_offset.
                            reduced_opening_for_log_height
                                .par_iter_mut()
                                .zip(mat_ros)
                                .for_each(|(ro, mat_ro)| {
                                    *ro += *alpha_offset * mat_ro;
                                });

                            // Update alpha_offset from α^i -> α^(i + 2 * width)
                            *alpha_offset *= alpha.exp_u64(2 * evals.values.width() as u64);
                        });
                });
            });

        // Iterate over our reduced columns and extract lambda - the multiple of the vanishing polynomial
        // which may appear in the reduced quotient due to CFFT dimension gap.

        let mut lambdas = vec![];
        let mut log_heights = vec![];
        let first_layer_mats: Vec<RowMajorMatrix<Challenge>> = reduced_openings
            .into_iter()
            .map(|(log_height, (_, mut ro))| {
                assert!(log_height > 0);
                log_heights.push(log_height);
                let lambda = extract_lambda(&mut ro, self.fri_params.log_blowup);
                lambdas.push(lambda);
                // Prepare for first layer fold with 2 siblings per leaf.
                RowMajorMatrix::new(ro, 2)
            })
            .collect();
        let log_max_height = log_heights.iter().max().copied().unwrap();

        // Commit to reduced openings at each log_height, so we can challenge a global
        // folding factor for all first layers, which we use for a "manual" (not part of p3-fri) fold.
        // This is necessary because the first layer of folding uses different twiddles, so it's easiest
        // to do it here, before p3-fri.

        let first_layer_dimensions = first_layer_mats
            .iter()
            .map(|matrix| matrix.dimensions())
            .collect::<Vec<_>>()
            .into();
        let (first_layer_commitment, first_layer_data) =
            self.fri_params.mmcs.commit(first_layer_mats);
        self.fri_params.mmcs.write_commitment(
            transcript,
            CircleLabelsDefault::first_layer_commitment(),
            first_layer_commitment,
        );
        let bivariate_beta = transcript
            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                CircleLabelsDefault::bivariate_beta(),
            )
            .into_inner();
        transcript
            .add_hints::<Challenge, ExtensionFieldCodec<Val, Challenge, FieldToFieldCodec<Val>>>(
                CircleLabelsDefault::lambdas(),
                &lambdas,
            );

        // Fold all first layers at bivariate_beta.

        let fri_input: Vec<Vec<Challenge>> = self
            .fri_params
            .mmcs
            .get_matrices(&first_layer_data)
            .into_iter()
            .map(|m| fold_y(bivariate_beta, m))
            // Reverse, because FRI expects descending by height
            .rev()
            .collect();

        prove(
            &CircleFriFolding,
            &self.fri_params,
            fri_input,
            transcript,
            |query, index, transcript| {
                // CircleFriFolder asks for an extra query index bit, so we use that here to index
                // the first layer fold.

                // Open the input (big opening, lots of columns) at the full index...
                for (batch, (data, _)) in rounds.iter().enumerate() {
                    let log_max_batch_height = log2_strict_usize(self.mmcs.get_max_height(data));
                    let reduced_index = index >> (log_max_height - log_max_batch_height);
                    let (opened_values, opening_proof) =
                        self.mmcs.open_batch(reduced_index, data).unpack();
                    let dimensions = self
                        .mmcs
                        .get_matrices(data)
                        .iter()
                        .map(|matrix| matrix.dimensions())
                        .collect::<Vec<_>>();
                    let batch_dimensions = BatchDimensions::from(dimensions.clone());
                    assert_eq!(opened_values.len(), dimensions.len());
                    for (matrix, (opened_values, dimensions)) in
                        opened_values.iter().zip(dimensions.iter()).enumerate()
                    {
                        assert_eq!(opened_values.len(), dimensions.width);
                        transcript.add_hints::<Val, FieldToFieldCodec<Val>>(
                            CircleLabelsDefault::input_opened_values(query, batch, matrix),
                            opened_values,
                        );
                    }
                    self.mmcs.write_proof_hint(
                        transcript,
                        CircleLabelsDefault::input_opening_proof(query, batch),
                        &batch_dimensions,
                        opening_proof,
                    );
                }

                // We committed to first_layer in pairs, so open the reduced index and include the sibling
                // as part of the input proof.
                let (first_layer_values, first_layer_proof) = self
                    .fri_params
                    .mmcs
                    .open_batch(index >> 1, &first_layer_data)
                    .unpack();
                let first_layer_siblings = izip!(&first_layer_values, &log_heights)
                    .map(|(v, log_height)| {
                        let reduced_index = index >> (log_max_height - log_height);
                        let sibling_index = (reduced_index & 1) ^ 1;
                        v[sibling_index]
                    })
                    .collect::<Vec<_>>();
                transcript.add_hints::<
                    Challenge,
                    ExtensionFieldCodec<Val, Challenge, FieldToFieldCodec<Val>>,
                >(
                    CircleLabelsDefault::first_layer_sibling_values(query),
                    &first_layer_siblings,
                );
                self.fri_params.mmcs.write_proof_hint(
                    transcript,
                    CircleLabelsDefault::first_layer_opening_proof(query),
                    &first_layer_dimensions,
                    first_layer_proof,
                );
            },
        );
    }

    fn verify(
        &self,
        rounds: Vec<CommitmentWithOpeningPoints<Challenge, Self::Commitment, Self::Domain>>,
        transcript: &mut VerifierState<'_, Challenger>,
    ) -> Result<(), Self::Error> {
        let batches = rounds
            .iter()
            .map(|(_, matrices)| BatchSpec::from_matrices(matrices))
            .collect::<Vec<_>>();

        let protocol = CircleProtocol::new(
            &self.fri_params,
            batches,
            CircleFriFolding::extra_query_index_bits(),
        );

        let alpha = transcript
            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                CircleLabelsDefault::alpha(),
            )
            .into_inner();
        let first_layer_commitment = self
            .fri_params
            .mmcs
            .read_commitment(
                transcript,
                CircleLabelsDefault::first_layer_commitment(),
                &protocol.first_layer_dimensions,
            )?
            .into_inner();
        let bivariate_beta = transcript
            .challenge_extension::<Val, Challenge, FieldToFieldCodec<Val>>(
                CircleLabelsDefault::bivariate_beta(),
            )
            .into_inner();
        let lambdas = transcript
            .next_hints::<Challenge, ExtensionFieldCodec<Val, Challenge, FieldToFieldCodec<Val>>>(
                CircleLabelsDefault::lambdas(),
                protocol.first_layer_dimensions.len(),
            )?;

        verify(
            &CircleFriFolding,
            &self.fri_params,
            &protocol,
            transcript,
            |query, index, transcript| {
                // log_height -> (alpha_offset, ro)
                let mut reduced_openings = BTreeMap::new();

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
                                    CircleLabelsDefault::input_opened_values(query, batch, matrix),
                                    dimensions.width,
                                )
                            })
                            .collect::<Result<Vec<_>, TranscriptError>>()?;
                        let opening_proof = self.mmcs.read_opening_proof(
                            transcript,
                            CircleLabelsDefault::input_opening_proof(query, batch),
                            dimensions,
                        )?;
                        Ok(BatchOpening::new(opened_values, opening_proof))
                    })
                    .collect::<Result<Vec<_>, TranscriptError>>()?;

                for (batch_opening, (batch_commit, mats)) in
                    zip_eq(input_openings, &rounds, InputError::InputShapeError)
                        .map_err(FriError::InputError)?
                {
                    let batch_heights: Vec<usize> = mats
                        .iter()
                        .map(|(domain, _)| domain.size() << self.fri_params.log_blowup)
                        .collect_vec();
                    let batch_dims: Vec<Dimensions> = mats
                        .iter()
                        .map(|(domain, points)| Dimensions {
                            width: points.first().map(|(_, values)| values.len()).unwrap_or(0),
                            height: domain.size() << self.fri_params.log_blowup,
                        })
                        .collect_vec();

                    let (dims, idx) = batch_heights
                        .iter()
                        .max()
                        .map(|x| log2_strict_usize(*x))
                        .map_or_else(
                            ||
                            // Empty batch?
                            (&[][..], 0),
                            |log_batch_max_height| {
                                (
                                    &batch_dims[..],
                                    index >> (protocol.query_bits() - log_batch_max_height),
                                )
                            },
                        );

                    self.mmcs
                        .verify_batch(batch_commit, dims, idx, (&batch_opening).into())
                        .map_err(|err| FriError::InputError(InputError::InputMmcsError(err)))?;

                    for (ps_at_x, (mat_domain, mat_points_and_values)) in zip_eq(
                        &batch_opening.opened_values,
                        mats,
                        InputError::InputShapeError,
                    )
                    .map_err(FriError::InputError)?
                    {
                        let log_height = mat_domain.log_n + self.fri_params.log_blowup;
                        let bits_reduced = protocol.query_bits() - log_height;
                        let orig_idx = cfft_permute_index(index >> bits_reduced, log_height);

                        let committed_domain = CircleDomain::standard(log_height);
                        let x = committed_domain.nth_point(orig_idx);

                        let (alpha_offset, ro) = reduced_openings
                            .entry(log_height)
                            .or_insert((Challenge::ONE, Challenge::ZERO));
                        let alpha_pow_width_2 = alpha.exp_u64(ps_at_x.len() as u64).square();

                        for (zeta_uni, ps_at_zeta) in mat_points_and_values {
                            let zeta = Point::from_projective_line(*zeta_uni);

                            *ro += *alpha_offset
                                * deep_quotient_reduce_row(alpha, x, zeta, ps_at_x, ps_at_zeta);

                            *alpha_offset *= alpha_pow_width_2;
                        }
                    }
                }

                // Verify bivariate fold and lambda correction
                let first_layer_siblings = transcript.next_hints::<Challenge, ExtensionFieldCodec<
                    Val,
                    Challenge,
                    FieldToFieldCodec<Val>,
                >>(
                    CircleLabelsDefault::first_layer_sibling_values(query),
                    protocol.first_layer_dimensions.len(),
                )?;
                let first_layer_proof = self.fri_params.mmcs.read_opening_proof(
                    transcript,
                    CircleLabelsDefault::first_layer_opening_proof(query),
                    &protocol.first_layer_dimensions,
                )?;

                let (mut fri_input, fl_dims, fl_leaves): (Vec<_>, Vec<_>, Vec<_>) = zip_eq(
                    zip_eq(
                        reduced_openings,
                        first_layer_siblings,
                        InputError::InputShapeError,
                    )
                    .map_err(FriError::InputError)?,
                    &lambdas,
                    InputError::InputShapeError,
                )
                .map_err(FriError::InputError)?
                .map(|(((log_height, (_, ro)), fl_sib), &lambda)| {
                    assert!(log_height > 0);

                    let orig_size = log_height - self.fri_params.log_blowup;
                    let bits_reduced = protocol.query_bits() - log_height;
                    let orig_idx = cfft_permute_index(index >> bits_reduced, log_height);

                    let lde_domain = CircleDomain::standard(log_height);
                    let p: Point<Val> = lde_domain.nth_point(orig_idx);

                    let lambda_corrected = ro - lambda * p.v_n(orig_size);

                    let mut fl_values = vec![lambda_corrected; 2];
                    fl_values[((index >> bits_reduced) & 1) ^ 1] = fl_sib;

                    let fri_input = (
                        // - 1 here is because we have already folded a layer.
                        log_height - 1,
                        fold_y_row(
                            index >> (bits_reduced + 1),
                            // - 1 here is log_arity.
                            log_height - 1,
                            bivariate_beta,
                            fl_values.iter().copied(),
                        ),
                    );

                    let fl_dims = Dimensions {
                        width: 0,
                        height: 1 << (log_height - 1),
                    };

                    (fri_input, fl_dims, fl_values)
                })
                .multiunzip();

                // sort descending
                fri_input.reverse();

                self.fri_params
                    .mmcs
                    .verify_batch(
                        &first_layer_commitment,
                        &fl_dims,
                        index >> 1,
                        BatchOpeningRef::new(&fl_leaves, &first_layer_proof),
                    )
                    .map_err(|err| FriError::InputError(InputError::FirstLayerMmcsError(err)))?;

                Ok(fri_input)
            },
        )
    }
}

impl<Val, InputMmcs, FriMmcs> BuildPeriodicLdeTableFast for CirclePcs<Val, InputMmcs, FriMmcs>
where
    Val: ComplexExtendable,
    InputMmcs: Mmcs<Val>,
{
    type PeriodicDomain = CircleDomain<Val>;

    fn maybe_build_periodic_lde_table_fast(
        &self,
        periodic_cols: &[Vec<p3_commit::Val<Self::PeriodicDomain>>],
        trace_domain: Self::PeriodicDomain,
        quotient_domain: Self::PeriodicDomain,
    ) -> Option<PeriodicLdeTable<p3_commit::Val<Self::PeriodicDomain>>>
    where
        p3_commit::Val<Self::PeriodicDomain>: Clone,
    {
        let table = build_periodic_lde_table_circle(periodic_cols, &trace_domain, &quotient_domain);
        Some(table)
    }
}
