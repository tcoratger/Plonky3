//! Transcript-shape construction for Circle PCS and Circle-FRI.
//!
//! Circle PCS has a small protocol layer before the FRI folding proof: input
//! openings are reduced, committed in width-2 first-layer matrices, folded with
//! a bivariate challenge, then passed into Circle-FRI. The types here describe
//! that verifier-known structure and append the ordered transcript interactions
//! consumed by prover and verifier states.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use p3_challenger::fs::{Hierarchy, Interaction, Kind, Label, Length};
use p3_commit::{BatchDimensions, MmcsTranscript};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_fri::FriParameters;
use p3_fri::protocol::ProtocolRounds;
pub use p3_fri::protocol::{BatchSpec, FriLabels, MatrixSpec};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;

/// Label scheme used by Circle PCS transcript construction.
///
/// This extends the regular FRI label scheme with Circle's first-layer
/// commitment, bivariate folding challenge, lambda hints, and first-layer
/// opening hints.
pub trait CircleLabels: FriLabels {
    /// Label for the commitment to Circle's width-2 first-layer matrices.
    const FIRST_LAYER_COMMITMENT: Label = "first-layer-commitment";
    /// Label for the challenge used to fold first-layer rows.
    const BIVARIATE_BETA: Label = "bivariate-beta";
    /// Label for lambda correction values from Circle quotient reduction.
    const LAMBDAS: Label = "lambdas";
    /// Prefix for first-layer sibling-value hints.
    const FIRST_LAYER_SIBLING_VALUES: Label = "first-layer-sibling-values";
    /// Prefix for first-layer MMCS opening-proof hints.
    const FIRST_LAYER_OPENING_PROOF: Label = "first-layer-opening-proof";

    /// Label for the first-layer commitment.
    fn first_layer_commitment() -> Label {
        Self::FIRST_LAYER_COMMITMENT
    }

    /// Label for the bivariate folding challenge.
    fn bivariate_beta() -> Label {
        Self::BIVARIATE_BETA
    }

    /// Label for the vector of lambda correction hints.
    fn lambdas() -> Label {
        Self::LAMBDAS
    }

    /// Label for first-layer sibling values read during `query`.
    fn first_layer_sibling_values(query: usize) -> Label {
        alloc::format!("{}-{query}", Self::FIRST_LAYER_SIBLING_VALUES).leak()
    }

    /// Label for the first-layer opening proof read during `query`.
    fn first_layer_opening_proof(query: usize) -> Label {
        alloc::format!("{}-{query}", Self::FIRST_LAYER_OPENING_PROOF).leak()
    }
}

/// Default Circle PCS transcript label scheme.
pub struct CircleLabelsDefault;

impl FriLabels for CircleLabelsDefault {
    const PROTOCOL: Label = "circle-fri";
}

impl CircleLabels for CircleLabelsDefault {}

/// Verifier-known shape of a full Circle-PCS-plus-Circle-FRI transcript.
///
/// `batches` describe the original Circle PCS inputs. The constructor derives
/// the input commitment dimensions, the first-layer commitment dimensions, and
/// the Circle-FRI folding-round schedule from those shapes and the FRI
/// parameters.
#[derive(Clone, Debug)]
pub struct CircleProtocol {
    /// Input PCS batches in commit/open order.
    pub batches: Vec<BatchSpec>,
    /// Blowup-domain MMCS dimensions for each input matrix in each batch.
    pub input_batch_dimensions: Vec<BatchDimensions>,
    /// MMCS dimensions for Circle's width-2 first-layer commitment.
    pub first_layer_dimensions: BatchDimensions,
    /// Log height of the largest Circle-FRI input after the first layer.
    pub log_max_height: usize,
    /// Extra query-index bits consumed by Circle's input/first-layer callback.
    pub extra_query_index_bits: usize,
    /// Derived Circle-FRI folding-round schedule.
    pub rounds: ProtocolRounds,
}

impl CircleProtocol {
    /// Derive a transcript shape for a Circle PCS run.
    ///
    /// `batches` describe base-domain inputs. The constructor first applies
    /// `params.log_blowup` to get input commitment dimensions, then accounts
    /// for Circle's first width-2 layer by halving the heights before replaying
    /// the FRI round schedule.
    pub fn new<M>(
        params: &FriParameters<M>,
        batches: Vec<BatchSpec>,
        extra_query_index_bits: usize,
    ) -> Self {
        <Self as p3_fri::protocol::Protocol>::new(params, batches, extra_query_index_bits, None)
    }

    /// Number of challenge bits sampled for each Circle-FRI query index.
    pub fn query_bits(&self) -> usize {
        self.log_max_height + self.extra_query_index_bits
    }
}

impl p3_fri::protocol::Protocol for CircleProtocol {
    fn new<M>(
        params: &FriParameters<M>,
        batches: Vec<BatchSpec>,
        extra_query_index_bits: usize,
        num_random_codewords: Option<usize>,
    ) -> Self {
        assert!(
            num_random_codewords.is_none(),
            "CircleProtocol does not support hiding random codewords",
        );
        assert_eq!(
            params.log_final_poly_len, 0,
            "Circle PCS currently requires log_final_poly_len = 0",
        );
        let input_batch_dimensions = batches
            .iter()
            .map(|batch| batch.blowup_dimensions(params.log_blowup))
            .collect::<Vec<_>>();
        let input_lengths = input_batch_dimensions
            .iter()
            .flat_map(|dimensions| dimensions.iter())
            .map(|dims| dims.height)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();
        let first_layer_log_heights = input_lengths
            .iter()
            .rev()
            .map(|&height| log2_strict_usize(height))
            .collect::<Vec<_>>();
        let first_layer_dimensions = first_layer_log_heights
            .iter()
            .map(|&log_height| Dimensions {
                width: 2,
                height: 1 << (log_height - 1),
            })
            .collect::<Vec<_>>()
            .into();
        let half_input_lengths = input_lengths
            .iter()
            .map(|&height| height >> 1)
            .collect::<Vec<_>>();
        let rounds = params.rounds(&half_input_lengths);
        let log_max_height = params.log_blowup + rounds.log_arities.iter().sum::<usize>();
        debug_assert_eq!(log_max_height, log2_strict_usize(half_input_lengths[0]));

        Self {
            batches,
            input_batch_dimensions,
            first_layer_dimensions,
            log_max_height,
            extra_query_index_bits,
            rounds,
        }
    }

    fn batches(&self) -> &[BatchSpec] {
        &self.batches
    }

    fn input_batch_dimensions(&self) -> &[BatchDimensions] {
        &self.input_batch_dimensions
    }

    fn log_max_height(&self) -> usize {
        self.log_max_height
    }

    fn extra_query_index_bits(&self) -> usize {
        self.extra_query_index_bits
    }

    fn rounds(&self) -> &ProtocolRounds {
        &self.rounds
    }

    fn append_pcs_interactions<Val, Challenge, InputMmcs, FriMmcs>(
        &self,
        interactions: &mut Vec<Interaction>,
        input_mmcs: &InputMmcs,
        params: &FriParameters<FriMmcs>,
    ) where
        Val: PrimeCharacteristicRing + Send + Sync + Clone,
        Challenge: BasedVectorSpace<Val> + Send + Sync + Clone,
        InputMmcs: MmcsTranscript<Val>,
        FriMmcs: MmcsTranscript<Challenge>,
    {
        for (batch_index, batch) in self.batches.iter().enumerate() {
            input_mmcs.append_commitment(
                interactions,
                CircleLabelsDefault::commitment(batch_index),
                &self.input_batch_dimensions[batch_index],
            );

            for (matrix_index, matrix) in batch.matrices().iter().enumerate() {
                for point_index in 0..matrix.num_opening_points() {
                    interactions.push(Interaction::new::<Challenge>(
                        Hierarchy::Atomic,
                        Kind::Challenge,
                        CircleLabelsDefault::opening_point(batch_index, matrix_index, point_index),
                        Length::Scalar,
                    ));
                }
            }
        }

        for (batch_index, batch) in self.batches.iter().enumerate() {
            for (matrix_index, matrix) in batch.matrices().iter().enumerate() {
                for point_index in 0..matrix.num_opening_points() {
                    interactions.push(Interaction::new::<Challenge>(
                        Hierarchy::Atomic,
                        Kind::Message,
                        CircleLabelsDefault::opened_values(batch_index, matrix_index, point_index),
                        Length::Fixed(matrix.dimensions().width),
                    ));
                }
            }
        }

        interactions.push(Interaction::new::<Challenge>(
            Hierarchy::Atomic,
            Kind::Challenge,
            CircleLabelsDefault::alpha(),
            Length::Scalar,
        ));
        params.mmcs.append_commitment(
            interactions,
            CircleLabelsDefault::first_layer_commitment(),
            &self.first_layer_dimensions,
        );
        interactions.push(Interaction::new::<Challenge>(
            Hierarchy::Atomic,
            Kind::Challenge,
            CircleLabelsDefault::bivariate_beta(),
            Length::Scalar,
        ));
        interactions.push(Interaction::new::<Challenge>(
            Hierarchy::Atomic,
            Kind::Hint,
            CircleLabelsDefault::lambdas(),
            Length::Fixed(self.first_layer_dimensions.len()),
        ));
    }

    fn append_fri_pattern<Val, Challenge, InputMmcs, FriMmcs>(
        &self,
        interactions: &mut Vec<Interaction>,
        params: &FriParameters<FriMmcs>,
        input_mmcs: &InputMmcs,
    ) where
        Val: PrimeCharacteristicRing + Send + Sync + Clone,
        Challenge: BasedVectorSpace<Val> + Send + Sync + Clone,
        InputMmcs: MmcsTranscript<Val>,
        FriMmcs: MmcsTranscript<Challenge>,
    {
        interactions.push(Interaction::new::<()>(
            Hierarchy::Begin,
            Kind::Protocol,
            CircleLabelsDefault::PROTOCOL,
            Length::None,
        ));

        for (round, dims) in self.rounds.dimensions.iter().enumerate() {
            params.mmcs.append_commitment(
                interactions,
                CircleLabelsDefault::round_commitment(round),
                &BatchDimensions::single(*dims),
            );
            interactions.push(Interaction::new::<Challenge>(
                Hierarchy::Atomic,
                Kind::Challenge,
                CircleLabelsDefault::beta(round),
                Length::Scalar,
            ));
        }

        interactions.push(Interaction::new::<Challenge>(
            Hierarchy::Atomic,
            Kind::Message,
            CircleLabelsDefault::final_poly(),
            Length::Scalar,
        ));
        interactions.push(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Pow,
            CircleLabelsDefault::pow_query(params.query_proof_of_work_bits),
            Length::Scalar,
        ));

        for query in 0..params.num_queries {
            interactions.push(Interaction::new::<usize>(
                Hierarchy::Atomic,
                Kind::Challenge,
                CircleLabelsDefault::query_index(query),
                Length::Bits(self.query_bits()),
            ));
            for (batch, dimensions) in self.input_batch_dimensions.iter().enumerate() {
                for (matrix, dimension) in dimensions.iter().enumerate() {
                    interactions.push(Interaction::new::<Val>(
                        Hierarchy::Atomic,
                        Kind::Hint,
                        CircleLabelsDefault::input_opened_values(query, batch, matrix),
                        Length::Fixed(dimension.width),
                    ));
                }
                input_mmcs.append_opening_proof_hint(
                    interactions,
                    CircleLabelsDefault::input_opening_proof(query, batch),
                    dimensions,
                );
            }
            interactions.push(Interaction::new::<Challenge>(
                Hierarchy::Atomic,
                Kind::Hint,
                CircleLabelsDefault::first_layer_sibling_values(query),
                Length::Fixed(self.first_layer_dimensions.len()),
            ));
            params.mmcs.append_opening_proof_hint(
                interactions,
                CircleLabelsDefault::first_layer_opening_proof(query),
                &self.first_layer_dimensions,
            );
            for (round, (&log_arity, dimensions)) in self
                .rounds
                .log_arities
                .iter()
                .zip(self.rounds.dimensions.iter())
                .enumerate()
            {
                interactions.push(Interaction::new::<Challenge>(
                    Hierarchy::Atomic,
                    Kind::Hint,
                    CircleLabelsDefault::sibling_values(query, round),
                    Length::Fixed(dimensions.width - 1),
                ));
                params.mmcs.append_opening_proof_hint(
                    interactions,
                    CircleLabelsDefault::round_opening_proof(query, round),
                    &BatchDimensions::single(*dimensions),
                );
                debug_assert_eq!(dimensions.width, 1 << log_arity);
            }
        }

        interactions.push(Interaction::new::<()>(
            Hierarchy::End,
            Kind::Protocol,
            CircleLabelsDefault::PROTOCOL,
            Length::None,
        ));
    }
}
