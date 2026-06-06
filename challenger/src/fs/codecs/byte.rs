//! Identity codec for byte-shaped sponges.

use crate::fs::codecs::Codec;
use crate::{CanObserve, CanSample};

/// Identity codec for challengers whose native alphabet is `u8`.
pub struct ByteCodec;

impl<C> Codec<C, u8> for ByteCodec
where
    C: CanObserve<u8> + CanSample<u8>,
{
    const SECURITY_BITS: u32 = 8;

    fn observe(challenger: &mut C, value: &u8) {
        challenger.observe(*value);
    }

    fn sample(challenger: &mut C) -> u8 {
        challenger.sample()
    }
}
