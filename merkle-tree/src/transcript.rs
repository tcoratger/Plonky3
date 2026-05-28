use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use itertools::Itertools;
use p3_challenger::fs::{
    DefaultCodec, Hierarchy, Interaction, Kind, Label, Length, ProverState, TranscriptBound,
    TranscriptError, VerifierState,
};
use p3_commit::{BatchDimensions, Mmcs, MmcsReader, MmcsTranscript, MmcsWriter};
use p3_field::PackedValue;
use p3_matrix::Dimensions;

use crate::MerkleTreeError::{EmptyBatch, IncompatibleHeights, IndexOutOfBounds};
use crate::merkle_tree::{padded_len, select_arity_step};
use crate::{MerkleCap, MerkleTreeError, MerkleTreeMmcs};

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize>
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
where
    P: PackedValue,
    PW: PackedValue,
{
    /// Number of Merkle sibling digests carried by an opening for these dimensions.
    pub fn opening_proof_len(&self, dimensions: &[Dimensions]) -> Result<usize, MerkleTreeError> {
        let arity_schedule = self.arity_schedule_for_dimensions(dimensions)?;
        let effective_cap_height = self.cap_height().min(arity_schedule.len());
        let proof_levels = arity_schedule.len().saturating_sub(effective_cap_height);
        Ok(arity_schedule[..proof_levels]
            .iter()
            .map(|step| step - 1)
            .sum())
    }

    pub(crate) fn commitment_roots_len(
        &self,
        dimensions: &[Dimensions],
    ) -> Result<usize, MerkleTreeError> {
        let arity_schedule = self.arity_schedule_for_dimensions(dimensions)?;
        let effective_cap_height = self.cap_height().min(arity_schedule.len());
        Ok(1 << effective_cap_height)
    }

    fn arity_schedule_for_dimensions(
        &self,
        dimensions: &[Dimensions],
    ) -> Result<Vec<usize>, MerkleTreeError> {
        if dimensions.is_empty() {
            return Err(EmptyBatch);
        }

        let mut heights_tallest_first = dimensions
            .iter()
            .map(|dims| dims.height)
            .sorted_by_key(|height| Reverse(*height))
            .peekable();

        if !heights_tallest_first
            .clone()
            .tuple_windows()
            .all(|(curr, next)| {
                curr == next || curr.next_power_of_two() != next.next_power_of_two()
            })
        {
            return Err(IncompatibleHeights);
        }

        let Some(&max_height) = heights_tallest_first.peek() else {
            return Err(EmptyBatch);
        };
        if max_height == 0 {
            return Err(IndexOutOfBounds {
                max_height,
                index: 0,
            });
        }

        let leaf_height_npt = max_height.next_power_of_two();
        heights_tallest_first
            .peeking_take_while(|height| *height == max_height)
            .for_each(|_| {});

        let mut curr_height_padded = padded_len(max_height, N);
        let mut arity_schedule = Vec::new();
        while curr_height_padded > 1 {
            let step = select_arity_step::<N>(
                curr_height_padded,
                leaf_height_npt,
                heights_tallest_first.clone(),
            );
            arity_schedule.push(step);

            let logical_next = curr_height_padded / step;
            curr_height_padded = padded_len(logical_next, N);

            let logical_next_npt = logical_next.next_power_of_two();
            let next_height = heights_tallest_first
                .peek()
                .copied()
                .filter(|height| height.next_power_of_two() == logical_next_npt);
            if let Some(next_height) = next_height {
                heights_tallest_first
                    .peeking_take_while(|height| *height == next_height)
                    .for_each(|_| {});
            }
        }

        Ok(arity_schedule)
    }
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize> MmcsTranscript<P::Value>
    for MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
where
    P: PackedValue,
    PW: PackedValue,
    PW::Value: Send + Sync + Clone,
    Self: Mmcs<
            P::Value,
            Commitment = MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>,
            Error = MerkleTreeError,
            Proof = Vec<[PW::Value; DIGEST_ELEMS]>,
        >,
{
    type Digest = PW::Value;

    fn append_commitment(
        &self,
        interactions: &mut Vec<Interaction>,
        label: Label,
        dimensions: &BatchDimensions,
    ) {
        let cap_roots_len = self
            .commitment_roots_len(dimensions.as_slice())
            .expect("valid Merkle commitment dimensions");
        interactions.push(Interaction::new::<PW::Value>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(cap_roots_len * DIGEST_ELEMS),
        ));
    }

    fn append_opening_proof_hint(
        &self,
        interactions: &mut Vec<Interaction>,
        label: Label,
        dimensions: &BatchDimensions,
    ) {
        let proof_len = self
            .opening_proof_len(dimensions.as_slice())
            .expect("valid Merkle opening dimensions");
        interactions.push(Interaction::new::<PW::Value>(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Fixed(proof_len * DIGEST_ELEMS),
        ));
    }
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize> MmcsWriter<P::Value>
    for MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
where
    P: PackedValue,
    PW: PackedValue,
    PW::Value: Send + Sync + Clone,
    Self: Mmcs<
            P::Value,
            Commitment = MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>,
            Error = MerkleTreeError,
            Proof = Vec<[PW::Value; DIGEST_ELEMS]>,
        >,
{
    fn write_commitment<Ch>(
        &self,
        transcript: &mut ProverState<Ch>,
        label: Label,
        commitment: Self::Commitment,
    ) where
        Ch: DefaultCodec<Self::Digest>,
    {
        let values = commitment
            .roots()
            .iter()
            .flat_map(|digest| digest.iter().cloned())
            .collect::<Vec<_>>();
        transcript
            .add_messages::<PW::Value, <Ch as DefaultCodec<Self::Digest>>::Codec>(label, &values);
    }

    fn write_proof_hint<Ch>(
        &self,
        transcript: &mut ProverState<Ch>,
        opening_proof_label: Label,
        dimensions: &BatchDimensions,
        opening_proof: Self::Proof,
    ) where
        Ch: DefaultCodec<Self::Digest>,
    {
        let proof_len = self
            .opening_proof_len(dimensions.as_slice())
            .expect("prover dimensions produce a valid opening shape");
        assert_eq!(opening_proof.len(), proof_len);
        let proof_values = opening_proof
            .iter()
            .flat_map(|digest| digest.iter().cloned())
            .collect::<Vec<_>>();
        transcript.add_hints::<PW::Value, <Ch as DefaultCodec<Self::Digest>>::Codec>(
            opening_proof_label,
            &proof_values,
        );
    }
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize> MmcsReader<P::Value>
    for MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
where
    P: PackedValue,
    PW: PackedValue,
    Self: Mmcs<
            P::Value,
            Commitment = MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>,
            Error = MerkleTreeError,
            Proof = Vec<[PW::Value; DIGEST_ELEMS]>,
        >,
{
    fn read_commitment<'a, Ch>(
        &self,
        transcript: &mut VerifierState<'a, Ch>,
        label: Label,
        dimensions: &BatchDimensions,
    ) -> Result<TranscriptBound<Self::Commitment>, TranscriptError>
    where
        Ch: DefaultCodec<Self::Digest>,
    {
        let cap_roots_len = self
            .commitment_roots_len(dimensions.as_slice())
            .map_err(|_| TranscriptError::BadProofShape {
                reason: "invalid Merkle commitment dimensions",
            })?;
        let bound_values = transcript
            .next_messages::<PW::Value, <Ch as DefaultCodec<Self::Digest>>::Codec>(
                label,
                cap_roots_len * DIGEST_ELEMS,
            )?;
        assert!(
            DIGEST_ELEMS > 0,
            "Merkle digest must contain at least one word"
        );
        assert!(
            !bound_values.is_empty() && bound_values.len() % DIGEST_ELEMS == 0,
            "Merkle cap transcript value count must be a non-empty multiple of DIGEST_ELEMS"
        );
        let mut bound_values = bound_values.into_iter();
        let first = bound_values
            .next()
            .expect("cannot bind an empty list of transcript values");
        Ok(bound_values
            .fold(first.map(|value| vec![value]), |acc, value| {
                acc.combine_with(value, |mut acc, value| {
                    acc.push(value);
                    acc
                })
            })
            .map(|values| {
                let roots = values
                    .chunks_exact(DIGEST_ELEMS)
                    .map(|chunk| core::array::from_fn(|i| chunk[i].clone()))
                    .collect();
                MerkleCap::new(roots)
            }))
    }

    fn read_opening_proof<'a, Ch>(
        &self,
        transcript: &mut VerifierState<'a, Ch>,
        opening_proof_label: Label,
        dimensions: &BatchDimensions,
    ) -> Result<Self::Proof, TranscriptError>
    where
        Ch: DefaultCodec<Self::Digest>,
    {
        let proof_len = self.opening_proof_len(dimensions.as_slice()).map_err(|_| {
            TranscriptError::BadProofShape {
                reason: "invalid Merkle opening dimensions",
            }
        })?;
        let proof_values = transcript
            .next_hints::<PW::Value, <Ch as DefaultCodec<Self::Digest>>::Codec>(
                opening_proof_label,
                proof_len * DIGEST_ELEMS,
            )?;
        split_digest_path(proof_values)
    }
}

fn split_digest_path<W: Clone, const DIGEST_ELEMS: usize>(
    values: Vec<W>,
) -> Result<Vec<[W; DIGEST_ELEMS]>, TranscriptError> {
    assert!(
        DIGEST_ELEMS > 0,
        "Merkle digest must contain at least one word"
    );

    if values.len() % DIGEST_ELEMS != 0 {
        return Err(TranscriptError::BadProofShape {
            reason: "Merkle opening proof value length is not digest-aligned",
        });
    }

    values
        .chunks_exact(DIGEST_ELEMS)
        .map(|words| {
            words
                .to_vec()
                .try_into()
                .map_err(|_| TranscriptError::BadProofShape {
                    reason: "Merkle opening proof digest has the wrong number of words",
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_challenger::fs::{
        DomainSeparator, FieldToFieldCodec, InteractionPattern, ProverState, VerifierState,
    };
    use p3_commit::{BatchDimensions, BatchOpening, Mmcs};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;
    type Cdc = FieldToFieldCodec<F>;

    fn mmcs() -> MyMmcs {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        MyMmcs::new(hash, compress, 0)
    }

    fn dimensions() -> BatchDimensions {
        vec![Dimensions {
            width: 1,
            height: 8,
        }]
        .into()
    }

    fn matrix() -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(vec![
            F::from_u32(1),
            F::from_u32(2),
            F::from_u32(3),
            F::from_u32(5),
            F::from_u32(8),
            F::from_u32(13),
            F::from_u32(21),
            F::from_u32(34),
        ])
    }

    fn domain_separator(mmcs: &MyMmcs) -> DomainSeparator<u8> {
        let mut interactions = Vec::new();
        mmcs.append_commitment(&mut interactions, "commitment", &dimensions());
        interactions.push(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Hint,
            "opened-values",
            Length::Fixed(dimensions().opened_values_len()),
        ));
        mmcs.append_opening_proof_hint(&mut interactions, "opening-proof", &dimensions());
        let pattern = InteractionPattern::new(interactions).unwrap();
        let mut ds = DomainSeparator::new(1, b"merkle-transcript-test", pattern);
        ds.bind_pattern_hash();
        ds
    }

    #[test]
    fn transcript_helpers_round_trip_merkle_opening() {
        let mmcs = mmcs();
        let dimensions = dimensions();
        let ds = domain_separator(&mmcs);
        let index = 3;
        let challenger = {
            let mut rng = SmallRng::seed_from_u64(2);
            Challenger::new(Perm::new_from_rng_128(&mut rng))
        };

        let (commitment, prover_data) = mmcs.commit_matrix(matrix());
        let (opened_values, opening_proof) = Mmcs::open_batch(&mmcs, index, &prover_data).unpack();

        let mut prover = ProverState::<_, u8>::new(challenger.clone(), &ds);
        mmcs.write_commitment(&mut prover, "commitment", commitment);
        assert_eq!(opened_values.len(), 1);
        prover.add_hints::<F, Cdc>("opened-values", &opened_values[0]);
        mmcs.write_proof_hint(&mut prover, "opening-proof", &dimensions, opening_proof);
        let narg = prover.finalize();

        let mut verifier = VerifierState::<_, u8>::new(challenger, &ds, &narg);
        let commitment = mmcs
            .read_commitment(&mut verifier, "commitment", &dimensions)
            .unwrap();
        let opened_values = verifier
            .next_hints::<F, Cdc>("opened-values", dimensions[0].width)
            .unwrap();
        let opening_proof = mmcs
            .read_opening_proof(&mut verifier, "opening-proof", &dimensions)
            .unwrap();
        let opening = BatchOpening::new(vec![opened_values], opening_proof);

        mmcs.verify_batch(commitment.as_inner(), &dimensions, index, (&opening).into())
            .unwrap();
        verifier.finalize().unwrap();
    }
}
