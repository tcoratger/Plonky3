//! Stateless adapter between values and a challenger.

use alloc::vec::Vec;

use crate::fs::error::TranscriptError;

/// Stateless absorb/sample adapter for values of type `T` against challenger `C`.
///
/// Codecs are zero-sized types picked at the call site.
///
/// One transcript may invoke several codecs in different roles.
pub trait Codec<C, T> {
    /// Bits of statistical security: `-log2` distance from uniform on `T`.
    const SECURITY_BITS: u32;

    /// Absorb `value` into the challenger.
    fn observe(challenger: &mut C, value: &T);

    /// Sample a fresh value from the challenger.
    fn sample(challenger: &mut C) -> T;

    /// Number of wire bytes used by one encoded `T`.
    fn byte_len() -> usize;

    /// Encode one value into the transcript wire format.
    fn encode(value: &T) -> Vec<u8>;

    /// Decode one value from the transcript wire format.
    fn decode(bytes: &[u8]) -> Result<T, TranscriptError>;
}
