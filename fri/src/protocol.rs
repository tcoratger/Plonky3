//! Transcript-shape construction for FRI and FRI-backed PCS protocols.
//!
//! The types in this module describe verifier-known protocol structure: input
//! matrix shapes, opening counts, FRI folding rounds, transcript labels, and
//! the ordered interaction pattern consumed by prover and verifier states.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
#[cfg(test)]
use core::ops::Range;

use itertools::Itertools;
use p3_challenger::fs::{Hierarchy, Interaction, Kind, Label, Length};
use p3_commit::{BatchDimensions, MmcsTranscript, PolynomialSpace};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;
#[cfg(test)]
use rand::{Rng, RngExt};

use crate::{FriParameters, compute_log_arity_for_round};

/// Label scheme used by FRI transcript construction.
///
/// Implement this trait to override label prefixes while preserving the same
/// indexed label layout.
pub trait FriLabels {
    /// Name of the FRI sub-protocol block.
    const PROTOCOL: Label = "fri";
    /// Prefix for input PCS commitments.
    const COMMITMENT: Label = "commitment";
    /// Prefix for FRI folding-round commitments.
    const ROUND_COMMITMENT: Label = "commitment-folding";
    /// Prefix for PCS opening-point challenges.
    const OPENING_POINT: Label = "opening-point";
    /// Prefix for PCS opened-value messages.
    const OPENED_VALUES: Label = "opened-values";
    /// Label for the final folded polynomial.
    const FINAL_POLY: Label = "final-poly";
    /// Label for the PCS batching challenge.
    const ALPHA: Label = "alpha";
    /// Label for the checked FRI arity schedule.
    const LOG_ARITIES: Label = "log-arities";
    /// Prefix for FRI folding challenges.
    const BETA: Label = "beta";
    /// Prefix for query-phase proof-of-work labels.
    const POW_QUERY: Label = "pow-query";
    /// Prefix for commit-phase proof-of-work labels.
    const POW_ROUND: Label = "pow-round";
    /// Prefix for FRI query-index challenges.
    const QUERY_INDEX: Label = "query-index";
    /// Prefix for queried input opened-value hints.
    const INPUT_OPENED_VALUES: Label = "input-opened-values";
    /// Prefix for queried input opening-proof hints.
    const INPUT_OPENING_PROOF: Label = "input-opening-proof";
    /// Prefix for FRI sibling-value hints opened during queries.
    const SIBLING_VALUES: Label = "sibling-values";
    /// Prefix for FRI round opening-proof hints.
    const ROUND_OPENING_PROOF: Label = "round-opening-proof";

    /// Label for the input PCS commitment of `batch`.
    fn commitment(batch: usize) -> Label {
        alloc::format!("{}-{batch}", Self::COMMITMENT).leak()
    }

    /// Label for the FRI folding commitment of `round`.
    fn round_commitment(round: usize) -> Label {
        alloc::format!("{}-{round}", Self::ROUND_COMMITMENT).leak()
    }

    /// Label for the `point`-th opening challenge of a matrix in a batch.
    fn opening_point(batch: usize, matrix: usize, point: usize) -> Label {
        alloc::format!("{}-{batch}-{matrix}-{point}", Self::OPENING_POINT).leak()
    }

    /// Label for opened values at one PCS opening point.
    fn opened_values(batch: usize, matrix: usize, point: usize) -> Label {
        alloc::format!("{}-{batch}-{matrix}-{point}", Self::OPENED_VALUES).leak()
    }

    /// Label for the final folded polynomial message.
    fn final_poly() -> Label {
        Self::FINAL_POLY
    }

    /// Label for the PCS batching challenge.
    fn alpha() -> Label {
        Self::ALPHA
    }

    /// Label for the checked FRI arity schedule.
    fn log_arities() -> Label {
        Self::LOG_ARITIES
    }

    /// Label for the folding challenge sampled after `round`.
    fn beta(round: usize) -> Label {
        alloc::format!("{}-{round}", Self::BETA).leak()
    }

    /// Label for query-phase proof-of-work with `bits` difficulty.
    fn pow_query(bits: usize) -> Label {
        alloc::format!("{}-{bits}", Self::POW_QUERY).leak()
    }

    /// Label for commit-phase proof-of-work with `bits` difficulty after `round`.
    fn pow_round(bits: usize, round: usize) -> Label {
        alloc::format!("{}-{bits}-{round}", Self::POW_ROUND).leak()
    }

    /// Label for the `query`-th query-index challenge.
    fn query_index(query: usize) -> Label {
        alloc::format!("{}-{query}", Self::QUERY_INDEX).leak()
    }

    /// Label for input opened values read during `query` for one matrix in a batch.
    fn input_opened_values(query: usize, batch: usize, matrix: usize) -> Label {
        alloc::format!("{}-{query}-{batch}-{matrix}", Self::INPUT_OPENED_VALUES).leak()
    }

    /// Label for the input opening proof read during `query` for `batch`.
    fn input_opening_proof(query: usize, batch: usize) -> Label {
        alloc::format!("{}-{query}-{batch}", Self::INPUT_OPENING_PROOF).leak()
    }

    /// Label for sibling values read during `query` for a FRI `round`.
    fn sibling_values(query: usize, round: usize) -> Label {
        alloc::format!("{}-{query}-{round}", Self::SIBLING_VALUES).leak()
    }

    /// Label for the FRI round opening proof read during `query`.
    fn round_opening_proof(query: usize, round: usize) -> Label {
        alloc::format!("{}-{query}-{round}", Self::ROUND_OPENING_PROOF).leak()
    }
}

/// Default FRI transcript label scheme.
pub struct FriLabelsDefault;

impl FriLabels for FriLabelsDefault {}

/// Public shape API for PCS protocols backed by the FRI transcript.
///
/// Implementors provide the verifier-known input shape, FRI round schedule, and
/// ordered transcript-pattern construction.
pub trait Protocol {
    /// Derive a protocol shape from verifier-known PCS inputs.
    ///
    /// `num_random_codewords` enables the hiding PCS shape for protocols that
    /// support it. Protocols without hiding support should reject `Some(_)`.
    fn new<M>(
        params: &FriParameters<M>,
        batches: Vec<BatchSpec>,
        extra_query_index_bits: usize,
        num_random_codewords: Option<usize>,
    ) -> Self;

    /// Input PCS batches in commit/open order.
    fn batches(&self) -> &[BatchSpec];

    /// Input-domain MMCS dimensions for each matrix in each input batch.
    fn input_batch_dimensions(&self) -> &[BatchDimensions];

    /// Log height of the largest FRI query domain.
    fn log_max_height(&self) -> usize;

    /// Extra query-index bits consumed by the PCS before FRI verification.
    fn extra_query_index_bits(&self) -> usize;

    /// Derived FRI folding-round schedule.
    fn rounds(&self) -> &ProtocolRounds;

    /// Number of challenge bits sampled for each query index.
    fn query_bits(&self) -> usize {
        self.log_max_height() + self.extra_query_index_bits()
    }

    /// Append the PCS input portion of the transcript pattern.
    fn append_pcs_interactions<Val, Challenge, InputMmcs, FriMmcs>(
        &self,
        interactions: &mut Vec<Interaction>,
        input_mmcs: &InputMmcs,
        params: &FriParameters<FriMmcs>,
    ) where
        Val: PrimeCharacteristicRing + Send + Sync + Clone,
        Challenge: BasedVectorSpace<Val> + Send + Sync + Clone,
        InputMmcs: MmcsTranscript<Val>,
        FriMmcs: MmcsTranscript<Challenge>;

    /// Append the FRI commit and query phases to the transcript pattern.
    fn append_fri_pattern<Val, Challenge, InputMmcs, FriMmcs>(
        &self,
        interactions: &mut Vec<Interaction>,
        params: &FriParameters<FriMmcs>,
        input_mmcs: &InputMmcs,
    ) where
        Val: PrimeCharacteristicRing + Send + Sync + Clone,
        Challenge: BasedVectorSpace<Val> + Send + Sync + Clone,
        InputMmcs: MmcsTranscript<Val>,
        FriMmcs: MmcsTranscript<Challenge>;
}

/// FRI folding-round schedule derived from the protocol input shapes.
///
/// Each entry describes one commit-phase round. `log_arities[i]` is the
/// log size of the folding arity used in round `i`, and
/// `dimensions[i]` is the shape of the MMCS matrix committed for that same
/// round.
#[derive(Clone, Debug)]
pub struct ProtocolRounds {
    /// Log size of the folding arity for each FRI round.
    pub log_arities: Vec<usize>,
    /// MMCS matrix dimensions committed by each FRI round.
    pub dimensions: Vec<Dimensions>,
}

impl<M> FriParameters<M> {
    /// Replays the FRI commit-phase schedule from verifier-known input heights.
    ///
    /// The returned dimensions are the matrices committed by each FRI round: one row
    /// per folded coset and one column per value inside that coset. When a later
    /// input has exactly the folded height, the original prover rolls it into the
    /// current folded vector; this helper mirrors that length transition without
    /// needing the actual values.
    pub fn rounds(&self, input_lengths: &[usize]) -> ProtocolRounds {
        assert!(!input_lengths.is_empty(), "FRI input lengths are empty");
        assert!(
            input_lengths.windows(2).all(|w| w[0] >= w[1]),
            "FRI input lengths must be sorted descending"
        );

        let mut input_lengths = input_lengths.iter().copied().peekable();
        let mut folded_len = input_lengths.next().unwrap();
        let mut log_arities = Vec::new();
        let mut dimensions = Vec::new();
        let log_final_height = self.log_blowup + self.log_final_poly_len;

        while folded_len > self.blowup() * self.final_poly_len() {
            let log_current_height = log2_strict_usize(folded_len);
            let next_input_log_height = input_lengths.peek().map(|&v| log2_strict_usize(v));
            let log_arity = compute_log_arity_for_round(
                log_current_height,
                next_input_log_height,
                log_final_height,
                self.max_log_arity,
            );
            let arity = 1 << log_arity;
            let dims = Dimensions {
                width: arity,
                height: folded_len >> log_arity,
            };

            log_arities.push(log_arity);
            dimensions.push(dims);

            folded_len >>= log_arity;
            if input_lengths.next_if_eq(&folded_len).is_some() {
                // The original prover rolls this next input into the folded vector,
                // leaving the current length unchanged for the next round.
            }
        }

        ProtocolRounds {
            log_arities,
            dimensions,
        }
    }
}

/// Verifier-known shape of a full PCS-plus-FRI transcript.
#[derive(Clone, Debug)]
pub struct FriProtocol {
    /// Input PCS batches in commit/open order.
    pub batches: Vec<BatchSpec>,
    /// Input-domain MMCS dimensions for each matrix in each input batch.
    pub input_batch_dimensions: Vec<BatchDimensions>,
    /// Log height of the largest input blowup domain.
    pub log_max_height: usize,
    /// Extra query-index bits consumed by the PCS before FRI query verification.
    pub extra_query_index_bits: usize,
    /// Number of random codeword columns appended by hiding PCS, if enabled.
    pub num_random_codewords: Option<usize>,
    /// Derived FRI folding-round schedule.
    pub rounds: ProtocolRounds,
}

/// Shape of one input PCS commitment batch.
#[derive(Clone, Debug)]
pub struct BatchSpec {
    matrices: Vec<MatrixSpec>,
}

impl BatchSpec {
    /// Build a batch from matrix shapes in commitment order.
    pub fn new(matrices: Vec<MatrixSpec>) -> Self {
        Self { matrices }
    }

    #[cfg(test)]
    pub(crate) fn random<R: Rng>(
        rng: &mut R,
        log_domain_sizes: &[usize],
        width_range: Range<usize>,
        num_opening_points_range: Range<usize>,
    ) -> Self {
        Self {
            matrices: log_domain_sizes
                .iter()
                .map(|&log_domain_size| {
                    MatrixSpec::random(
                        rng,
                        log_domain_size,
                        width_range.clone(),
                        num_opening_points_range.clone(),
                    )
                })
                .collect(),
        }
    }

    /// Matrix shapes in this batch.
    pub fn matrices(&self) -> &[MatrixSpec] {
        &self.matrices
    }

    /// MMCS dimensions after applying the public FRI blowup factor.
    pub fn blowup_dimensions(&self, log_blowup: usize) -> BatchDimensions {
        self.matrices
            .iter()
            .map(|matrix| matrix.blowup_dimensions(log_blowup))
            .collect::<Vec<_>>()
            .into()
    }

    /// MMCS dimensions for hiding PCS inputs.
    ///
    /// Hiding PCS appends random codeword columns and commits on the doubled
    /// hiding domain.
    pub fn hiding_blowup_dimensions(
        &self,
        log_blowup: usize,
        num_random_codewords: usize,
    ) -> BatchDimensions {
        self.matrices
            .iter()
            .map(|matrix| matrix.hiding_blowup_dimensions(log_blowup, num_random_codewords))
            .collect::<Vec<_>>()
            .into()
    }

    /// Infer a batch shape from verifier claims.
    pub fn from_matrices<Challenge, Domain>(
        matrices: &[(Domain, Vec<(Challenge, Vec<Challenge>)>)],
    ) -> Self
    where
        Domain: PolynomialSpace,
    {
        Self::new(
            matrices
                .iter()
                .map(|(domain, points)| {
                    let width = points
                        .iter()
                        .map(|(_, values)| values.len())
                        .all_equal_value()
                        .unwrap();
                    MatrixSpec::new(
                        p3_matrix::Dimensions {
                            width,
                            height: domain.size(),
                        },
                        points.len(),
                    )
                })
                .collect(),
        )
    }
}

/// Shape of one committed matrix and its requested opening count.
#[derive(Clone, Debug)]
pub struct MatrixSpec {
    dimensions: Dimensions,
    num_opening_points: usize,
}

impl MatrixSpec {
    /// Build a matrix specification from base-domain dimensions and opening count.
    pub fn new(dimensions: Dimensions, num_opening_points: usize) -> Self {
        Self {
            dimensions,
            num_opening_points,
        }
    }

    /// Base-domain height before FRI blowup.
    pub fn height(&self) -> usize {
        self.dimensions.height
    }

    /// Matrix width before any hiding columns are appended.
    pub fn width(&self) -> usize {
        self.dimensions.width
    }

    #[cfg(test)]
    pub(crate) fn random<R: Rng>(
        rng: &mut R,
        log_domain_size: usize,
        width_range: Range<usize>,
        num_opening_points_range: Range<usize>,
    ) -> Self {
        Self {
            dimensions: Dimensions {
                width: rng.random_range(width_range),
                height: 1 << log_domain_size,
            },
            num_opening_points: rng.random_range(num_opening_points_range),
        }
    }

    /// Base-domain matrix dimensions.
    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    /// Number of opening points sampled for this matrix.
    pub fn num_opening_points(&self) -> usize {
        self.num_opening_points
    }

    /// MMCS dimensions after applying the public FRI blowup factor.
    pub fn blowup_dimensions(&self, log_blowup: usize) -> Dimensions {
        Dimensions {
            width: self.dimensions.width,
            height: self.dimensions.height << log_blowup,
        }
    }

    /// MMCS dimensions for a hiding PCS commitment of this matrix.
    pub fn hiding_blowup_dimensions(
        &self,
        log_blowup: usize,
        num_random_codewords: usize,
    ) -> Dimensions {
        Dimensions {
            width: self.dimensions.width + num_random_codewords,
            height: self.dimensions.height << (log_blowup + 1),
        }
    }
}

impl FriProtocol {
    /// Derive a transcript shape for a non-hiding FRI PCS run.
    ///
    /// `batches` describe the base-domain inputs. The constructor applies
    /// `params.log_blowup`, derives query width from the largest input domain,
    /// and computes the FRI folding-round schedule.
    pub fn new<M>(
        params: &FriParameters<M>,
        batches: Vec<BatchSpec>,
        extra_query_index_bits: usize,
    ) -> Self {
        <Self as Protocol>::new(params, batches, extra_query_index_bits, None)
    }

    /// Derive a transcript shape for a hiding FRI PCS run.
    ///
    /// This uses hiding commitment dimensions: each matrix gets
    /// `num_random_codewords` extra columns and commits on the doubled hiding
    /// domain.
    pub fn new_hiding<M>(
        params: &FriParameters<M>,
        batches: Vec<BatchSpec>,
        extra_query_index_bits: usize,
        num_random_codewords: usize,
    ) -> Self {
        <Self as Protocol>::new(
            params,
            batches,
            extra_query_index_bits,
            Some(num_random_codewords),
        )
    }

    /// Number of challenge bits sampled for each query index.
    pub fn query_bits(&self) -> usize {
        self.log_max_height + self.extra_query_index_bits
    }
}

impl Protocol for FriProtocol {
    fn new<M>(
        params: &FriParameters<M>,
        batches: Vec<BatchSpec>,
        extra_query_index_bits: usize,
        num_random_codewords: Option<usize>,
    ) -> Self {
        let input_batch_dimensions = batches
            .iter()
            .map(|batch| {
                if let Some(num_random_codewords) = num_random_codewords {
                    batch.hiding_blowup_dimensions(params.log_blowup, num_random_codewords)
                } else {
                    batch.blowup_dimensions(params.log_blowup)
                }
            })
            .collect::<Vec<_>>();
        let input_lengths = input_batch_dimensions
            .iter()
            .flat_map(|dimensions| dimensions.iter())
            .map(|dims| dims.height)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();
        let rounds = params.rounds(&input_lengths);
        let log_max_height = params.log_blowup
            + params.log_final_poly_len
            + rounds.log_arities.iter().sum::<usize>();
        debug_assert_eq!(log_max_height, log2_strict_usize(input_lengths[0]));

        Self {
            batches,
            input_batch_dimensions,
            log_max_height,
            extra_query_index_bits,
            num_random_codewords,
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
        _params: &FriParameters<FriMmcs>,
    ) where
        Val: PrimeCharacteristicRing + Send + Sync + Clone,
        Challenge: BasedVectorSpace<Val> + Send + Sync + Clone,
        InputMmcs: MmcsTranscript<Val>,
        FriMmcs: MmcsTranscript<Challenge>,
    {
        for (batch_index, batch) in self.batches.iter().enumerate() {
            input_mmcs.append_commitment(
                interactions,
                FriLabelsDefault::commitment(batch_index),
                &self.input_batch_dimensions[batch_index],
            );

            for (matrix_index, matrix) in batch.matrices.iter().enumerate() {
                for point_index in 0..matrix.num_opening_points {
                    interactions.push(Interaction::new::<Challenge>(
                        Hierarchy::Atomic,
                        Kind::Challenge,
                        FriLabelsDefault::opening_point(batch_index, matrix_index, point_index),
                        Length::Scalar,
                    ));
                }
            }
        }

        for (batch_index, batch) in self.batches.iter().enumerate() {
            for (matrix_index, matrix) in batch.matrices.iter().enumerate() {
                for point_index in 0..matrix.num_opening_points {
                    let opened_values_len =
                        matrix.dimensions.width + self.num_random_codewords.unwrap_or(0);
                    interactions.push(Interaction::new::<Challenge>(
                        Hierarchy::Atomic,
                        Kind::Message,
                        FriLabelsDefault::opened_values(batch_index, matrix_index, point_index),
                        Length::Fixed(opened_values_len),
                    ));
                }
            }
        }

        interactions.push(Interaction::new::<Challenge>(
            Hierarchy::Atomic,
            Kind::Challenge,
            FriLabelsDefault::alpha(),
            Length::Scalar,
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
            FriLabelsDefault::PROTOCOL,
            Length::None,
        ));

        for (round, dims) in self.rounds.dimensions.iter().enumerate() {
            params.mmcs.append_commitment(
                interactions,
                FriLabelsDefault::round_commitment(round),
                &BatchDimensions::single(*dims),
            );
            interactions.push(Interaction::new::<u64>(
                Hierarchy::Atomic,
                Kind::Pow,
                FriLabelsDefault::pow_round(params.commit_proof_of_work_bits, round),
                Length::Scalar,
            ));
            interactions.push(Interaction::new::<Challenge>(
                Hierarchy::Atomic,
                Kind::Challenge,
                FriLabelsDefault::beta(round),
                Length::Scalar,
            ));
        }

        interactions.push(Interaction::new::<Challenge>(
            Hierarchy::Atomic,
            Kind::Message,
            FriLabelsDefault::final_poly(),
            Length::Fixed(params.final_poly_len()),
        ));
        interactions.push(Interaction::new::<Val>(
            Hierarchy::Atomic,
            Kind::Message,
            FriLabelsDefault::log_arities(),
            Length::Fixed(self.rounds.log_arities.len()),
        ));
        interactions.push(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Pow,
            FriLabelsDefault::pow_query(params.query_proof_of_work_bits),
            Length::Scalar,
        ));

        for query in 0..params.num_queries {
            interactions.push(Interaction::new::<usize>(
                Hierarchy::Atomic,
                Kind::Challenge,
                FriLabelsDefault::query_index(query),
                Length::Bits(self.query_bits()),
            ));
            for (batch, dimensions) in self.input_batch_dimensions.iter().enumerate() {
                for (matrix, dimension) in dimensions.iter().enumerate() {
                    interactions.push(Interaction::new::<Val>(
                        Hierarchy::Atomic,
                        Kind::Hint,
                        FriLabelsDefault::input_opened_values(query, batch, matrix),
                        Length::Fixed(dimension.width),
                    ));
                }
                input_mmcs.append_opening_proof_hint(
                    interactions,
                    FriLabelsDefault::input_opening_proof(query, batch),
                    dimensions,
                );
            }
            for (round, dimensions) in self.rounds.dimensions.iter().enumerate() {
                interactions.push(Interaction::new::<Challenge>(
                    Hierarchy::Atomic,
                    Kind::Hint,
                    FriLabelsDefault::sibling_values(query, round),
                    Length::Fixed(dimensions.width - 1),
                ));
                params.mmcs.append_opening_proof_hint(
                    interactions,
                    FriLabelsDefault::round_opening_proof(query, round),
                    &BatchDimensions::single(*dimensions),
                );
            }
        }

        interactions.push(Interaction::new::<()>(
            Hierarchy::End,
            Kind::Protocol,
            FriLabelsDefault::PROTOCOL,
            Length::None,
        ));
    }
}
