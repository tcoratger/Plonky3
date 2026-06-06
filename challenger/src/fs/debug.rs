//! Debug helpers for transcript inspection and proof mutation.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::any::type_name;

use p3_field::{BasedVectorSpace, PrimeField};

use crate::fs::ExtensionFieldCodec;
use crate::fs::codecs::Codec;
use crate::fs::codecs::decode_field::{
    decode_field_be_canonical, encode_field_be, field_byte_size,
};
use crate::fs::codecs::length_prefix::{bound_byte_width, decode_len_be};
use crate::fs::error::TranscriptError;
use crate::fs::pattern::{Interaction, Kind, Label, Length};

/// Label lookup mode for [`ProofDebug`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelUniqueness {
    /// Atomic labels must be unique, and label lookup does not take an occurrence index.
    Enforced,
    /// Duplicate atomic labels are allowed, and label lookup requires an occurrence index.
    Allowed,
}

/// Byte layout of one proof-carrying interaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofObjectInfo {
    /// Index of the interaction in the pattern.
    pub interaction_index: usize,
    /// Interaction metadata from the pattern.
    pub interaction: Interaction,
    /// Byte offset of the proof object inside the proof.
    pub byte_offset: usize,
}

/// Owned debug view over one proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofDebug {
    proof: Vec<u8>,
    proof_objects: Vec<ProofObjectInfo>,
    label_to_indices: BTreeMap<Label, Vec<usize>>,
    label_uniqueness: LabelUniqueness,
}

impl ProofDebug {
    pub(crate) fn new(
        proof: Vec<u8>,
        proof_objects: Vec<ProofObjectInfo>,
        label_uniqueness: LabelUniqueness,
    ) -> Self {
        let label_to_indices = build_label_to_indices(&proof_objects);
        if label_uniqueness == LabelUniqueness::Enforced {
            validate_unique_labels(&label_to_indices);
        }
        Self {
            proof,
            proof_objects,
            label_to_indices,
            label_uniqueness,
        }
    }

    /// Return the current proof bytes.
    #[must_use]
    pub fn proof(&self) -> &[u8] {
        &self.proof
    }

    /// Consume the debugger and return the proof bytes.
    #[must_use]
    pub fn into_proof(self) -> Vec<u8> {
        self.proof
    }

    fn proof_object_info(
        &self,
        label: Label,
        occurrence: Option<usize>,
    ) -> Result<ProofObjectInfo, TranscriptError> {
        let indices = self
            .label_to_indices
            .get(label)
            .ok_or(TranscriptError::BadProofShape {
                reason: "debug label occurrence not found",
            })?;
        let index = match (self.label_uniqueness, occurrence) {
            (LabelUniqueness::Enforced, None) => indices.first().copied(),
            (LabelUniqueness::Enforced, Some(_)) => {
                panic!("occurrence index requires non-unique label mode")
            }
            (LabelUniqueness::Allowed, Some(occurrence)) => indices.get(occurrence).copied(),
            (LabelUniqueness::Allowed, None) => {
                panic!("unique lookup requires enforced label mode")
            }
        }
        .ok_or(TranscriptError::BadProofShape {
            reason: "debug label occurrence not found",
        })?;
        Ok(self.proof_objects[index])
    }

    /// Decode, expose, and re-encode one message.
    ///
    /// The callback may only inspect the value or may mutate it.
    pub fn get_message<T, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut T),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let info = self.proof_object_info(label, occurrence)?;
        let interaction = info.interaction;
        expect_type::<T>(interaction)?;
        expect_kind(interaction, Kind::Message)?;
        expect_length(interaction.length(), Length::Scalar)?;
        self.get_one::<T, Cdc, C>(info, get)
    }

    /// Decode, expose, and re-encode a fixed-length message list.
    ///
    /// The callback may only inspect the values or may mutate them.
    pub fn get_messages<T, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut [T]),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let info = self.proof_object_info(label, occurrence)?;
        let interaction = info.interaction;
        expect_type::<T>(interaction)?;
        expect_kind(interaction, Kind::Message)?;
        let count = fixed_count(interaction.length())?;
        self.get_many::<T, Cdc, C>(info, count, get)
    }

    /// Decode, expose, and re-encode a bounded message list.
    ///
    /// The encoded bounded length is treated as metadata and cannot be changed.
    /// The callback may only inspect or mutate the already-encoded values.
    pub fn get_messages_bounded<T, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut [T]),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let info = self.proof_object_info(label, occurrence)?;
        let interaction = info.interaction;
        expect_type::<T>(interaction)?;
        expect_kind(interaction, Kind::Message)?;
        let max = bounded_max(interaction.length())?;
        self.get_bounded_many::<T, Cdc, C>(info, max, get)
    }

    /// Decode, expose, and re-encode one hint.
    ///
    /// The callback may only inspect the value or may mutate it.
    pub fn get_hint<T, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut T),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let info = self.proof_object_info(label, occurrence)?;
        let interaction = info.interaction;
        expect_type::<T>(interaction)?;
        expect_kind(interaction, Kind::Hint)?;
        expect_length(interaction.length(), Length::Scalar)?;
        self.get_one::<T, Cdc, C>(info, get)
    }

    /// Decode, expose, and re-encode a fixed-length hint list.
    ///
    /// The callback may only inspect the values or may mutate them.
    pub fn get_hints<T, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut [T]),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let info = self.proof_object_info(label, occurrence)?;
        let interaction = info.interaction;
        expect_type::<T>(interaction)?;
        expect_kind(interaction, Kind::Hint)?;
        let count = fixed_count(interaction.length())?;
        self.get_many::<T, Cdc, C>(info, count, get)
    }

    /// Decode, expose, and re-encode a bounded hint list.
    ///
    /// The encoded bounded length is treated as metadata and cannot be changed.
    /// The callback may only inspect or mutate the already-encoded values.
    pub fn get_hints_bounded<T, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut [T]),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let info = self.proof_object_info(label, occurrence)?;
        let interaction = info.interaction;
        expect_type::<T>(interaction)?;
        expect_kind(interaction, Kind::Hint)?;
        let max = bounded_max(interaction.length())?;
        self.get_bounded_many::<T, Cdc, C>(info, max, get)
    }

    /// Expose fixed-width salt bytes.
    ///
    /// The callback may only inspect the bytes or may mutate them.
    pub fn get_salt(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut [u8]),
    ) -> Result<(), TranscriptError> {
        let info = self.proof_object_info(label, occurrence)?;
        let interaction = info.interaction;
        expect_type::<u8>(interaction)?;
        expect_kind(interaction, Kind::Salt)?;
        let count = fixed_count(interaction.length())?;
        self.checked_range(info.byte_offset, count)?;
        get(&mut self.proof[info.byte_offset..info.byte_offset + count]);
        Ok(())
    }

    /// Decode, expose, and re-encode one proof-of-work witness.
    ///
    /// The callback may only inspect the witness or may mutate it.
    pub fn get_pow<Val>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut Val),
    ) -> Result<(), TranscriptError>
    where
        Val: PrimeField,
    {
        let info = self.proof_object_info(label, occurrence)?;
        let interaction = info.interaction;
        expect_kind(interaction, Kind::Pow)?;
        expect_length(interaction.length(), Length::Scalar)?;
        let width = field_byte_size::<Val>();
        self.checked_range(info.byte_offset, width)?;
        let mut value = decode_field_be_canonical::<Val>(
            &self.proof[info.byte_offset..info.byte_offset + width],
        )?;
        get(&mut value);
        let encoded = encode_field_be(&value);
        self.proof[info.byte_offset..info.byte_offset + width].copy_from_slice(&encoded);
        Ok(())
    }

    /// Decode, expose, and re-encode one scalar message.
    ///
    /// Convenience wrapper around [`Self::get_message`].
    pub fn get_scalar<F, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut F),
    ) -> Result<(), TranscriptError>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.get_message::<F, Cdc, C>(label, occurrence, get)
    }

    /// Decode, expose, and re-encode a fixed-length scalar message list.
    ///
    /// Convenience wrapper around [`Self::get_messages`].
    pub fn get_scalars<F, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut [F]),
    ) -> Result<(), TranscriptError>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.get_messages::<F, Cdc, C>(label, occurrence, get)
    }

    /// Decode, expose, and re-encode a bounded scalar message list.
    ///
    /// Convenience wrapper around [`Self::get_messages_bounded`].
    pub fn get_scalars_bounded<F, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut [F]),
    ) -> Result<(), TranscriptError>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.get_messages_bounded::<F, Cdc, C>(label, occurrence, get)
    }

    /// Decode, expose, and re-encode one extension-field message.
    ///
    /// Convenience wrapper around [`Self::get_message`].
    pub fn get_extension<F, EF, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut EF),
    ) -> Result<(), TranscriptError>
    where
        F: PrimeField,
        EF: BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        self.get_message::<EF, ExtensionFieldCodec<F, EF, Cdc>, C>(label, occurrence, get)
    }

    /// Decode, expose, and re-encode a fixed-length extension-field message list.
    ///
    /// Convenience wrapper around [`Self::get_messages`].
    pub fn get_extensions<F, EF, Cdc, C>(
        &mut self,
        label: Label,
        occurrence: Option<usize>,
        get: impl FnOnce(&mut [EF]),
    ) -> Result<(), TranscriptError>
    where
        F: PrimeField,
        EF: BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        self.get_messages::<EF, ExtensionFieldCodec<F, EF, Cdc>, C>(label, occurrence, get)
    }

    fn get_one<T, Cdc, C>(
        &mut self,
        info: ProofObjectInfo,
        get: impl FnOnce(&mut T),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let byte_len = Cdc::byte_len();
        self.checked_range(info.byte_offset, byte_len)?;
        let mut value = Cdc::decode(&self.proof[info.byte_offset..info.byte_offset + byte_len])?;
        get(&mut value);
        let encoded = Cdc::encode(&value);
        if encoded.len() != byte_len {
            return Err(TranscriptError::BadProofShape {
                reason: "debug mutation changed encoded byte length",
            });
        }
        self.proof[info.byte_offset..info.byte_offset + byte_len].copy_from_slice(&encoded);
        Ok(())
    }

    fn get_many<T, Cdc, C>(
        &mut self,
        info: ProofObjectInfo,
        count: usize,
        get: impl FnOnce(&mut [T]),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        self.get_many_at_offset::<T, Cdc, C>(info.byte_offset, count, get)
    }

    fn get_bounded_many<T, Cdc, C>(
        &mut self,
        info: ProofObjectInfo,
        max: usize,
        get: impl FnOnce(&mut [T]),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let width = bound_byte_width(max);
        self.checked_range(info.byte_offset, width)?;
        let actual = decode_len_be(
            &self.proof[info.byte_offset..info.byte_offset + width],
            width,
        );
        if actual > max {
            return Err(TranscriptError::BadProofShape {
                reason: "debug bounded length exceeds declared maximum",
            });
        }
        self.get_many_at_offset::<T, Cdc, C>(info.byte_offset + width, actual, get)
    }

    fn get_many_at_offset<T, Cdc, C>(
        &mut self,
        byte_offset: usize,
        count: usize,
        get: impl FnOnce(&mut [T]),
    ) -> Result<(), TranscriptError>
    where
        Cdc: Codec<C, T>,
    {
        let value_len = Cdc::byte_len();
        let byte_len = count * value_len;
        self.checked_range(byte_offset, byte_len)?;

        let mut values = Vec::with_capacity(count);
        for chunk in self.proof[byte_offset..byte_offset + byte_len].chunks_exact(value_len) {
            values.push(Cdc::decode(chunk)?);
        }

        get(&mut values);

        for (value, chunk) in values
            .iter()
            .zip(self.proof[byte_offset..byte_offset + byte_len].chunks_exact_mut(value_len))
        {
            let encoded = Cdc::encode(value);
            if encoded.len() != value_len {
                return Err(TranscriptError::BadProofShape {
                    reason: "debug mutation changed encoded byte length",
                });
            }
            chunk.copy_from_slice(&encoded);
        }
        Ok(())
    }

    const fn checked_range(
        &self,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<(), TranscriptError> {
        if byte_offset + byte_len > self.proof.len() {
            return Err(TranscriptError::BadProofShape {
                reason: "NARG ended before all expected bytes were read",
            });
        }
        Ok(())
    }
}

const fn bounded_max(length: Length) -> Result<usize, TranscriptError> {
    match length {
        Length::Bounded(max) => Ok(max),
        _ => Err(TranscriptError::BadProofShape {
            reason: "debug proof object is not bounded",
        }),
    }
}

fn validate_unique_labels(label_to_indices: &BTreeMap<Label, Vec<usize>>) {
    for (label, indices) in label_to_indices {
        assert!(
            indices.len() == 1,
            "duplicate proof-carrying transcript label: {label}"
        );
    }
}

fn build_label_to_indices(proof_objects: &[ProofObjectInfo]) -> BTreeMap<Label, Vec<usize>> {
    let mut label_to_indices = BTreeMap::<Label, Vec<usize>>::new();
    for (position, info) in proof_objects.iter().enumerate() {
        label_to_indices
            .entry(info.interaction.label())
            .or_default()
            .push(position);
    }
    label_to_indices
}

const fn fixed_count(length: Length) -> Result<usize, TranscriptError> {
    match length {
        Length::Scalar => Ok(1),
        Length::Fixed(n) => Ok(n),
        Length::Bounded(_) => Err(TranscriptError::BadProofShape {
            reason: "debug proof object is bounded; use a bounded getter",
        }),
        Length::Dynamic => Err(TranscriptError::BadProofShape {
            reason: "dynamic debug proof objects are not supported",
        }),
        Length::None => Err(TranscriptError::BadProofShape {
            reason: "debug proof object has no carried value",
        }),
    }
}

fn expect_kind(interaction: Interaction, kind: Kind) -> Result<(), TranscriptError> {
    if interaction.kind() == kind {
        Ok(())
    } else {
        Err(TranscriptError::BadProofShape {
            reason: "debug kind does not match interaction kind",
        })
    }
}

fn expect_length(actual: Length, expected: Length) -> Result<(), TranscriptError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TranscriptError::BadProofShape {
            reason: "debug length does not match interaction length",
        })
    }
}

fn expect_type<T>(interaction: Interaction) -> Result<(), TranscriptError> {
    if interaction.type_name() == type_name::<T>() {
        Ok(())
    } else {
        Err(TranscriptError::BadProofShape {
            reason: "debug type does not match interaction type",
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::fs::codecs::BytesToFieldCodec;
    use crate::fs::domain_separator::DomainSeparator;
    use crate::fs::pattern::{Hierarchy, Interaction, InteractionPattern, Kind, Length};
    use crate::fs::shake128::Shake128;
    use crate::fs::state::{ProverState, VerifierState};
    use crate::{CanObserve, CanSample};

    type F = BabyBear;
    type Cdc = BytesToFieldCodec<F>;

    /// Identity byte codec.
    struct ByteCodec;

    impl Codec<Shake128, u8> for ByteCodec {
        const SECURITY_BITS: u32 = 8;

        fn observe(challenger: &mut Shake128, value: &u8) {
            challenger.observe(*value);
        }

        fn sample(challenger: &mut Shake128) -> u8 {
            challenger.sample()
        }

        fn byte_len() -> usize {
            1
        }

        fn encode(value: &u8) -> Vec<u8> {
            vec![*value]
        }

        fn decode(bytes: &[u8]) -> Result<u8, TranscriptError> {
            match bytes {
                [value] => Ok(*value),
                _ => Err(TranscriptError::BadProofShape {
                    reason: "byte encoding must contain exactly one byte",
                }),
            }
        }
    }

    #[test]
    fn debug_get_checks_and_mutates_unique_proof_objects() {
        let pattern = InteractionPattern::new(vec![
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Message, "value", Length::Scalar),
            Interaction::new::<u8>(Hierarchy::Atomic, Kind::Hint, "hint", Length::Fixed(2)),
        ])
        .unwrap();
        let mut ds = DomainSeparator::<u8>::new(0, b"debug-proof", pattern);
        ds.bind_pattern_hash();

        let mut prover = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        prover.add_scalar::<F, Cdc>("value", &F::from_u32(3));
        prover.add_hints::<u8, ByteCodec>("hint", &[1, 2]);
        let mut proof_debug = prover.finalize_with_debug(LabelUniqueness::Enforced);
        proof_debug
            .get_scalar::<F, Cdc, Shake128>("value", None, |value| {
                assert_eq!(*value, F::from_u32(3));
                *value = F::from_u32(9);
            })
            .unwrap();
        proof_debug
            .get_hints::<u8, ByteCodec, Shake128>("hint", None, |bytes| {
                assert_eq!(bytes, &[1, 2]);
                bytes[1] = 7;
            })
            .unwrap();
        let proof = proof_debug.into_proof();

        let mut verifier = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &proof);
        let value = verifier.next_scalar::<F, Cdc>("value").unwrap();
        let hint = verifier.next_hint("hint", 2).unwrap();
        verifier.finalize().unwrap();
        assert_eq!(*value.as_inner(), F::from_u32(9));
        assert_eq!(hint, &[1, 7]);
    }

    #[test]
    fn debug_occurrence_lookup_mutates_duplicate_labels() {
        let pattern = InteractionPattern::new(vec![
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Message, "value", Length::Scalar),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Message, "value", Length::Scalar),
        ])
        .unwrap();
        let mut ds = DomainSeparator::<u8>::new(0, b"debug-dupes", pattern);
        ds.bind_pattern_hash();

        let mut prover = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        prover.add_scalar::<F, Cdc>("value", &F::from_u32(1));
        prover.add_scalar::<F, Cdc>("value", &F::from_u32(2));
        let mut proof_debug = prover.finalize_with_debug(LabelUniqueness::Allowed);
        proof_debug
            .get_scalar::<F, Cdc, Shake128>("value", Some(1), |value| {
                assert_eq!(*value, F::from_u32(2));
                *value = F::from_u32(5);
            })
            .unwrap();
        let proof = proof_debug.into_proof();

        let mut verifier = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &proof);
        let first = verifier.next_scalar::<F, Cdc>("value").unwrap();
        let second = verifier.next_scalar::<F, Cdc>("value").unwrap();
        verifier.finalize().unwrap();
        assert_eq!(*first.as_inner(), F::from_u32(1));
        assert_eq!(*second.as_inner(), F::from_u32(5));
    }

    #[test]
    fn debug_mutates_bounded_messages_without_changing_length() {
        let pattern = InteractionPattern::new(vec![Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Message,
            "values",
            Length::Bounded(5),
        )])
        .unwrap();
        let mut ds = DomainSeparator::<u8>::new(0, b"debug-bounded-msgs", pattern);
        ds.bind_pattern_hash();

        let mut prover = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        prover.add_scalars_bounded::<F, Cdc>("values", &[F::from_u32(3), F::from_u32(4)], 5);
        let mut proof_debug = prover.finalize_with_debug(LabelUniqueness::Enforced);
        proof_debug
            .get_scalars_bounded::<F, Cdc, Shake128>("values", None, |values| {
                assert_eq!(values, &[F::from_u32(3), F::from_u32(4)]);
                values[1] = F::from_u32(9);
            })
            .unwrap();
        let proof = proof_debug.into_proof();

        let mut verifier = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &proof);
        let values = verifier
            .next_scalars_bounded::<F, Cdc>("values", 5)
            .unwrap();
        verifier.finalize().unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(*values[0].as_inner(), F::from_u32(3));
        assert_eq!(*values[1].as_inner(), F::from_u32(9));
    }

    #[test]
    fn debug_mutates_bounded_hints_without_changing_length() {
        let pattern = InteractionPattern::new(vec![Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Hint,
            "auth",
            Length::Bounded(8),
        )])
        .unwrap();
        let mut ds = DomainSeparator::<u8>::new(0, b"debug-bounded-hints", pattern);
        ds.bind_pattern_hash();

        let mut prover = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        prover.add_hints_bounded::<u8, ByteCodec>("auth", &[0xaa, 0xbb, 0xcc], 8);
        let mut proof_debug = prover.finalize_with_debug(LabelUniqueness::Enforced);
        proof_debug
            .get_hints_bounded::<u8, ByteCodec, Shake128>("auth", None, |hints| {
                assert_eq!(hints, &[0xaa, 0xbb, 0xcc]);
                hints[0] = 0x11;
            })
            .unwrap();
        let proof = proof_debug.into_proof();

        let mut verifier = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &proof);
        let hints = verifier
            .next_hints_bounded::<u8, ByteCodec>("auth", 8)
            .unwrap();
        verifier.finalize().unwrap();
        assert_eq!(hints, &[0x11, 0xbb, 0xcc]);
    }

    #[test]
    #[should_panic(expected = "duplicate proof-carrying transcript label")]
    fn debug_enforced_uniqueness_rejects_duplicate_labels() {
        let pattern = InteractionPattern::new(vec![
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Message, "value", Length::Scalar),
            Interaction::new::<F>(Hierarchy::Atomic, Kind::Message, "value", Length::Scalar),
        ])
        .unwrap();
        let mut ds = DomainSeparator::<u8>::new(0, b"debug-dupes", pattern);
        ds.bind_pattern_hash();

        let mut prover = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        prover.add_scalar::<F, Cdc>("value", &F::from_u32(1));
        prover.add_scalar::<F, Cdc>("value", &F::from_u32(2));
        let _ = prover.finalize_with_debug(LabelUniqueness::Enforced);
    }
}
