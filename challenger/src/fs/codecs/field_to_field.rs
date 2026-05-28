//! Identity codec for sponges whose alphabet is the field itself.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::PrimeField;

use crate::fs::codecs::Codec;
use crate::fs::codecs::decode_field::{
    decode_field_be_canonical, encode_field_be, field_byte_size,
};
use crate::fs::error::TranscriptError;
use crate::{CanObserve, CanSample};

/// Identity codec: the sponge alphabet is the field, so no encoding step is needed.
#[derive(Clone, Copy, Debug, Default)]
pub struct FieldToFieldCodec<F>(PhantomData<F>);

impl<C, F> Codec<C, F> for FieldToFieldCodec<F>
where
    F: PrimeField,
    C: CanObserve<F> + CanSample<F>,
{
    const SECURITY_BITS: u32 = 128;

    fn observe(challenger: &mut C, value: &F) {
        challenger.observe(*value);
    }

    fn sample(challenger: &mut C) -> F {
        challenger.sample()
    }

    fn byte_len() -> usize {
        field_byte_size::<F>()
    }

    fn encode(value: &F) -> Vec<u8> {
        encode_field_be(value)
    }

    fn decode(bytes: &[u8]) -> Result<F, TranscriptError> {
        decode_field_be_canonical(bytes)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};

    use super::*;

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;

    /// FIFO sponge over `F`.
    #[derive(Default)]
    struct FieldFifo {
        queue: Vec<F>,
    }

    impl CanObserve<F> for FieldFifo {
        fn observe(&mut self, v: F) {
            self.queue.push(v);
        }
    }

    impl CanSample<F> for FieldFifo {
        fn sample(&mut self) -> F {
            self.queue.remove(0)
        }
    }

    #[test]
    fn observe_then_sample_round_trips() {
        // Identity codec: bytes-in = bytes-out, no transformation.
        let mut s = FieldFifo::default();

        let v1 = F::from_u32(7u32);
        let v2 = F::from_u32(123u32);
        let v3 = F::from_u32(F::ORDER_U32 - 1);

        FieldToFieldCodec::<F>::observe(&mut s, &v1);
        FieldToFieldCodec::<F>::observe(&mut s, &v2);
        FieldToFieldCodec::<F>::observe(&mut s, &v3);

        assert_eq!(FieldToFieldCodec::<F>::sample(&mut s), v1);
        assert_eq!(FieldToFieldCodec::<F>::sample(&mut s), v2);
        assert_eq!(FieldToFieldCodec::<F>::sample(&mut s), v3);
    }
}
