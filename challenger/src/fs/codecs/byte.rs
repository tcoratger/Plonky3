//! Identity byte codec.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{PrimeField32, PrimeField64};
use p3_symmetric::CryptographicHasher;

use crate::fs::codecs::Codec;
use crate::fs::error::TranscriptError;
use crate::fs::shake128::Shake128;
use crate::{
    CanObserve, CanSample, HashChallenger, SerializingChallenger32, SerializingChallenger64,
};

/// Codec for transcript values that are already bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteCodec;

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
        encode_byte(value)
    }

    fn decode(bytes: &[u8]) -> Result<u8, TranscriptError> {
        decode_byte(bytes)
    }
}

impl<H, const OUT_LEN: usize> Codec<HashChallenger<u8, H, OUT_LEN>, u8> for ByteCodec
where
    H: CryptographicHasher<u8, [u8; OUT_LEN]>,
{
    const SECURITY_BITS: u32 = 8;

    fn observe(challenger: &mut HashChallenger<u8, H, OUT_LEN>, value: &u8) {
        challenger.observe(*value);
    }

    fn sample(challenger: &mut HashChallenger<u8, H, OUT_LEN>) -> u8 {
        challenger.sample()
    }

    fn byte_len() -> usize {
        1
    }

    fn encode(value: &u8) -> Vec<u8> {
        encode_byte(value)
    }

    fn decode(bytes: &[u8]) -> Result<u8, TranscriptError> {
        decode_byte(bytes)
    }
}

impl<F, Inner> Codec<SerializingChallenger32<F, Inner>, u8> for ByteCodec
where
    F: PrimeField32,
    Inner: CanObserve<u8> + CanSample<u8>,
{
    const SECURITY_BITS: u32 = 8;

    fn observe(challenger: &mut SerializingChallenger32<F, Inner>, value: &u8) {
        challenger.inner.observe(*value);
    }

    fn sample(challenger: &mut SerializingChallenger32<F, Inner>) -> u8 {
        challenger.inner.sample()
    }

    fn byte_len() -> usize {
        1
    }

    fn encode(value: &u8) -> Vec<u8> {
        encode_byte(value)
    }

    fn decode(bytes: &[u8]) -> Result<u8, TranscriptError> {
        decode_byte(bytes)
    }
}

impl<F, Inner> Codec<SerializingChallenger64<F, Inner>, u8> for ByteCodec
where
    F: PrimeField64,
    Inner: CanObserve<u8> + CanSample<u8>,
{
    const SECURITY_BITS: u32 = 8;

    fn observe(challenger: &mut SerializingChallenger64<F, Inner>, value: &u8) {
        challenger.inner.observe(*value);
    }

    fn sample(challenger: &mut SerializingChallenger64<F, Inner>) -> u8 {
        challenger.inner.sample()
    }

    fn byte_len() -> usize {
        1
    }

    fn encode(value: &u8) -> Vec<u8> {
        encode_byte(value)
    }

    fn decode(bytes: &[u8]) -> Result<u8, TranscriptError> {
        decode_byte(bytes)
    }
}

fn encode_byte(value: &u8) -> Vec<u8> {
    vec![*value]
}

fn decode_byte(bytes: &[u8]) -> Result<u8, TranscriptError> {
    match bytes {
        [value] => Ok(*value),
        _ => Err(TranscriptError::BadProofShape {
            reason: "byte encoding must contain exactly one byte",
        }),
    }
}
