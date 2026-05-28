//! Verifier-side driver.

use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, Field, PrimeField};

use crate::fs::ExtensionFieldCodec;
use crate::fs::bound::TranscriptBound;
use crate::fs::codecs::Codec;
use crate::fs::codecs::decode_field::{
    decode_field_be_canonical, encode_field_be, field_byte_size,
};
use crate::fs::domain_separator::{CanObserveBytes, DomainSeparator};
use crate::fs::error::TranscriptError;
use crate::fs::pattern::{Hierarchy, Interaction, Kind, Label, Length, Pattern, PatternPlayer};
use crate::fs::unit::Unit;
use crate::{CanObserve, CanSampleBits, GrindingChallenger};

/// Drives a verifier-side transcript in lockstep with a recorded pattern.
///
/// Reads bytes from a caller-supplied slice through a cursor — no copies.
///
/// Wire-format problems return a structured error.
///
/// Pattern misuse panics with a diff message.
pub struct VerifierState<'a, C, U: Unit = u8> {
    /// Underlying sponge, seeded identically to the prover.
    challenger: C,
    /// Pattern player that validates each call against the recorded sequence.
    player: PatternPlayer,
    /// Caller-supplied wire bytes consumed in order.
    narg: &'a [u8],
    /// Read position into the wire bytes.
    cursor: usize,
    /// Type-level marker for the sponge alphabet.
    _u: core::marker::PhantomData<U>,
}

impl<C, U: Unit> Drop for VerifierState<'_, C, U> {
    fn drop(&mut self) {
        // Abort the player on drop when the user did not finalise
        //
        // This way, error paths do not turn into double-panics during cleanup.
        if !self.player.is_finalized() {
            self.player.abort();
        }
    }
}

impl<'a, C, U: Unit> VerifierState<'a, C, U> {
    /// Build a driver and seed the challenger from the domain separator.
    pub fn new(mut challenger: C, ds: &DomainSeparator<U>, narg: &'a [u8]) -> Self
    where
        C: CanObserveBytes,
    {
        // Seed identically to the prover so both sides land on the same sponge state.
        ds.seed_bytes(&mut challenger);
        let player = PatternPlayer::new(ds.pattern().clone());
        Self {
            challenger,
            player,
            narg,
            cursor: 0,
            _u: core::marker::PhantomData,
        }
    }

    /// Read-only access to the underlying challenger.
    pub const fn challenger(&self) -> &C {
        &self.challenger
    }

    /// Number of wire bytes still ahead of the cursor.
    pub const fn remaining_narg(&self) -> usize {
        self.narg.len() - self.cursor
    }

    /// Finalise the driver.
    ///
    /// # Errors
    ///
    /// Returns an error when wire bytes remain unread.
    ///
    /// # Panics
    ///
    /// When the recorded pattern is not fully replayed.
    pub fn finalize(self) -> Result<(), TranscriptError> {
        // Move fields out by hand: the wrapper's Drop must not run.
        let this = core::mem::ManuallyDrop::new(self);
        // SAFETY: each field is moved out exactly once and Drop never runs on `this`.
        let player = unsafe { core::ptr::read(&this.player) };
        let cursor = this.cursor;
        let narg_len = this.narg.len();
        let challenger = unsafe { core::ptr::read(&this.challenger) };
        drop(challenger);
        // Pattern check: every recorded step must have been replayed.
        player.finalize();
        // Wire check: trailing bytes mean the prover smuggled data the verifier never read.
        if cursor != narg_len {
            return Err(TranscriptError::BadProofShape {
                reason: "trailing NARG bytes after final verifier step",
            });
        }
        Ok(())
    }

    /// Take `n` raw bytes from the wire cursor, or fail if out of bounds.
    fn take_bytes(&mut self, n: usize) -> Result<&'a [u8], TranscriptError> {
        if self.cursor + n > self.narg.len() {
            return Err(TranscriptError::BadProofShape {
                reason: "NARG ended before all expected bytes were read",
            });
        }
        let slice = &self.narg[self.cursor..self.cursor + n];
        self.cursor += n;
        Ok(slice)
    }

    /// Replay a salt step by reading `byte_len` bytes from the wire.
    pub fn next_salt(&mut self, byte_len: usize) -> Result<Vec<u8>, TranscriptError>
    where
        C: CanObserve<u8>,
    {
        // The verifier must know the length up front:
        //
        // Reading it from the wire would let an attacker control how much data is consumed.
        self.player.interact(Interaction::new::<u8>(
            Hierarchy::Atomic,
            Kind::Salt,
            "salt",
            Length::Fixed(byte_len),
        ));
        let bytes = self.take_bytes(byte_len)?.to_vec();
        // Absorb so future samples depend on the prover's salt.
        self.challenger.observe_slice(&bytes);
        Ok(bytes)
    }

    /// Replay an `add_scalar` step from the prover.
    pub fn next_scalar<F, Cdc>(
        &mut self,
        label: Label,
    ) -> Result<TranscriptBound<F>, TranscriptError>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.next_message::<F, Cdc>(label)
    }

    /// Replay an `add_scalars` step from the prover.
    pub fn next_scalars<F, Cdc>(
        &mut self,
        label: Label,
        n: usize,
    ) -> Result<Vec<TranscriptBound<F>>, TranscriptError>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.next_messages::<F, Cdc>(label, n)
    }

    /// Replay an `add_message` step from the prover.
    pub fn next_message<T, Cdc>(
        &mut self,
        label: Label,
    ) -> Result<TranscriptBound<T>, TranscriptError>
    where
        T: Clone,
        Cdc: Codec<C, T>,
    {
        self.player.interact(Interaction::new::<T>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Scalar,
        ));
        let raw = self.take_bytes(Cdc::byte_len())?;
        let value = Cdc::decode(raw)?;
        Cdc::observe(&mut self.challenger, &value);
        Ok(TranscriptBound::wrap(value))
    }

    /// Replay an `add_messages` step from the prover.
    pub fn next_messages<T, Cdc>(
        &mut self,
        label: Label,
        n: usize,
    ) -> Result<Vec<TranscriptBound<T>>, TranscriptError>
    where
        T: Clone,
        Cdc: Codec<C, T>,
    {
        self.player.interact(Interaction::new::<T>(
            Hierarchy::Atomic,
            Kind::Message,
            label,
            Length::Fixed(n),
        ));
        let need = Cdc::byte_len();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let raw = self.take_bytes(need)?;
            let value = Cdc::decode(raw)?;
            Cdc::observe(&mut self.challenger, &value);
            out.push(TranscriptBound::wrap(value));
        }
        Ok(out)
    }

    /// Replay an `add_extension` step from the prover.
    pub fn next_extension<F, EF, Cdc>(
        &mut self,
        label: Label,
    ) -> Result<TranscriptBound<EF>, TranscriptError>
    where
        F: PrimeField,
        EF: Field + BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        self.next_message::<EF, ExtensionFieldCodec<F, EF, Cdc>>(label)
    }

    /// Replay an `add_extensions` step from the prover.
    pub fn next_extensions<F, EF, Cdc>(
        &mut self,
        label: Label,
        n: usize,
    ) -> Result<Vec<TranscriptBound<EF>>, TranscriptError>
    where
        F: PrimeField,
        EF: Field + BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        self.next_messages::<EF, ExtensionFieldCodec<F, EF, Cdc>>(label, n)
    }

    /// Replay an `add_hint` step.
    ///
    /// Hint values are returned to the caller; they are never absorbed.
    pub fn next_hint<T, Cdc>(&mut self, label: Label) -> Result<T, TranscriptError>
    where
        T: Clone,
        Cdc: Codec<C, T>,
    {
        self.player.interact(Interaction::new::<T>(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Scalar,
        ));
        let raw = self.take_bytes(Cdc::byte_len())?;
        Cdc::decode(raw)
    }

    /// Replay an `add_hints` step.
    pub fn next_hints<T, Cdc>(&mut self, label: Label, n: usize) -> Result<Vec<T>, TranscriptError>
    where
        T: Clone,
        Cdc: Codec<C, T>,
    {
        self.player.interact(Interaction::new::<T>(
            Hierarchy::Atomic,
            Kind::Hint,
            label,
            Length::Fixed(n),
        ));
        let need = Cdc::byte_len();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(Cdc::decode(self.take_bytes(need)?)?);
        }
        Ok(out)
    }

    /// Sample one challenge scalar in lockstep with the prover.
    pub fn challenge_scalar<F, Cdc>(&mut self, label: Label) -> TranscriptBound<F>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            Length::Scalar,
        ));
        TranscriptBound::wrap(Cdc::sample(&mut self.challenger))
    }

    /// Sample `n` challenge scalars in lockstep with the prover.
    pub fn challenge_scalars<F, Cdc>(&mut self, label: Label, n: usize) -> Vec<TranscriptBound<F>>
    where
        F: PrimeField,
        Cdc: Codec<C, F>,
    {
        self.player.interact(Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            Length::Fixed(n),
        ));
        (0..n)
            .map(|_| TranscriptBound::wrap(Cdc::sample(&mut self.challenger)))
            .collect()
    }

    /// Sample one challenge integer with a caller-supplied bit width.
    pub fn challenge_bits(&mut self, label: Label, bits: usize) -> TranscriptBound<usize>
    where
        C: CanSampleBits<usize>,
    {
        self.player.interact(Interaction::new::<usize>(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            Length::Bits(bits),
        ));
        TranscriptBound::wrap(self.challenger.sample_bits(bits))
    }

    /// Sample one challenge extension-field element in lockstep with the prover.
    pub fn challenge_extension<F, EF, Cdc>(&mut self, label: Label) -> TranscriptBound<EF>
    where
        F: PrimeField,
        EF: BasedVectorSpace<F>,
        Cdc: Codec<C, F>,
    {
        self.player.interact(Interaction::new::<EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            Length::Scalar,
        ));
        TranscriptBound::wrap(EF::from_basis_coefficients_fn(|_| {
            Cdc::sample(&mut self.challenger)
        }))
    }

    /// Replay a proof-of-work step.
    ///
    /// - Reads the witness from the wire,
    /// - Checks its encoding is canonical,
    /// - Absorbs it through the challenger's PoW path.
    pub fn check_pow(&mut self, label: Label, bits: usize) -> Result<(), TranscriptError>
    where
        C: GrindingChallenger,
        <C as GrindingChallenger>::Witness: PrimeField,
    {
        // Validate: the next pattern step is a proof-of-work step.
        self.player.interact(Interaction::new::<u64>(
            Hierarchy::Atomic,
            Kind::Pow,
            label,
            Length::Scalar,
        ));
        // Read the witness from the wire and decode it as a canonical field element.
        let need = field_byte_size::<<C as GrindingChallenger>::Witness>();
        let raw = self.take_bytes(need)?;
        let witness = decode_field_be_canonical::<<C as GrindingChallenger>::Witness>(raw)?;
        // Re-encode and compare so a malicious prover cannot smuggle a non-canonical encoding.
        let canonical = encode_field_be::<<C as GrindingChallenger>::Witness>(&witness);
        if canonical.as_slice() != raw {
            return Err(TranscriptError::BadProofShape {
                reason: "pow witness encoding is non-canonical",
            });
        }
        // Verify the witness produces the required number of leading zero bits.
        if !self.challenger.check_witness(bits, witness) {
            return Err(TranscriptError::BadProofShape {
                reason: "pow witness does not produce enough zero bits",
            });
        }
        Ok(())
    }

    /// Open a sub-protocol marker in the recorded pattern.
    pub fn begin_protocol<T: ?Sized>(&mut self, label: Label) {
        self.player.interact(Interaction::new::<T>(
            Hierarchy::Begin,
            Kind::Protocol,
            label,
            Length::None,
        ));
    }

    /// Close a sub-protocol marker in the recorded pattern.
    pub fn end_protocol<T: ?Sized>(&mut self, label: Label) {
        self.player.interact(Interaction::new::<T>(
            Hierarchy::End,
            Kind::Protocol,
            label,
            Length::None,
        ));
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::fs::codecs::BytesToFieldCodec;
    use crate::fs::pattern::InteractionPattern;
    use crate::fs::shake128::Shake128;

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;

    fn one_msg_pattern() -> InteractionPattern {
        InteractionPattern::new(vec![Interaction::new::<F>(
            Hierarchy::Atomic,
            Kind::Message,
            "msg",
            Length::Scalar,
        )])
        .unwrap()
    }

    #[test]
    fn truncated_narg_yields_bad_proof_shape() {
        // Pattern wants a 4-byte scalar; verifier gets 1 byte.
        let pat = one_msg_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"trunc", pat);
        ds.bind_pattern_hash();
        let narg = [0u8; 1];
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let err = v
            .next_scalar::<F, BytesToFieldCodec<F>>("msg")
            .expect_err("truncated NARG must error");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "NARG ended before all expected bytes were read",
            }
        );
    }

    #[test]
    fn non_canonical_scalar_encoding_is_rejected() {
        // 0xFFFFFFFF > BabyBear order, so canonical decoding rejects it.
        let pat = one_msg_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"non-canon", pat);
        ds.bind_pattern_hash();
        let narg = [0xffu8; 4];
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let err = v
            .next_scalar::<F, BytesToFieldCodec<F>>("msg")
            .expect_err("non-canonical encoding must error");
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "field encoding outside canonical range",
            }
        );
    }

    #[test]
    fn trailing_narg_bytes_rejected_at_finalize() {
        // Pattern: one scalar message.
        let pat = one_msg_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"trailing", pat);
        ds.bind_pattern_hash();

        // Prover writes a valid NARG, then we smuggle one extra byte at the tail.
        use crate::fs::state::ProverState;
        let mut p = ProverState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds);
        p.add_scalar::<F, BytesToFieldCodec<F>>("msg", &F::from_u32(7u32));
        let mut narg = p.finalize();
        narg.push(0x42);

        // Verifier consumes the legal scalar, then finalize must reject the leftover byte.
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let _ = v
            .next_scalar::<F, BytesToFieldCodec<F>>("msg")
            .expect("legal scalar");
        let err = v.finalize().expect_err("trailing bytes must be rejected");

        // Property: finalize reports the exact "trailing bytes" reason.
        assert_eq!(
            err,
            TranscriptError::BadProofShape {
                reason: "trailing NARG bytes after final verifier step",
            }
        );
    }

    #[test]
    #[should_panic(expected = "Received interaction")]
    fn pattern_mismatch_on_label_panics() {
        // Pattern declares "msg" but the caller asks for "different".
        let pat = one_msg_pattern();
        let mut ds: DomainSeparator<u8> = DomainSeparator::new(0, b"mismatch", pat);
        ds.bind_pattern_hash();
        let narg = [0u8; 4];
        let mut v = VerifierState::<_, u8>::new(Shake128::new(&[0u8; 64]), &ds, &narg);
        let _ = v.next_scalar::<F, BytesToFieldCodec<F>>("different");
    }
}
