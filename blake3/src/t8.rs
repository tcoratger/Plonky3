//! T8 leaves: records of eight 32-byte blocks hashed with three BLAKE3 compressions.
//!
//! A plain BLAKE3 hash of the same 256 bytes takes four.

#[cfg(test)]
mod tests;

use blake3::OUT_LEN;
use blake3::platform::Platform;
use p3_symmetric::CryptographicHasher;

use crate::batch::t8::{FLAGS, ROLES};

/// Bytes in one block, the digest width n = 256 bits.
pub const BLOCK_BYTES: usize = 32;

/// Bytes a stage adds after the first: seven blocks, `z_2` to `z_8`.
///
/// The first stage reads eight blocks, so a record of `k` stages is `32 + 224 k` bytes.
pub const STAGE_STRIDE: usize = 7 * BLOCK_BYTES;

/// Words of a chaining value, or of one block.
type Words = [u32; 8];

/// Stages in a record of `len` bytes.
///
/// Returns `None` unless `len = 32 + 224 k` with `k >= 1`.
#[must_use]
pub const fn stages(len: usize) -> Option<usize> {
    match len.checked_sub(BLOCK_BYTES) {
        Some(rest) if rest >= STAGE_STRIDE && rest.is_multiple_of(STAGE_STRIDE) => {
            Some(rest / STAGE_STRIDE)
        }
        _ => None,
    }
}

/// The T8 leaf hash of Dodis, Khovratovich, Mouha and Nandi, on BLAKE3 compressions.
///
/// One stage hashes eight blocks with three calls:
///
/// ```text
///     a = h_1(z_1; z_2 z_3)
///     b = h_2(z_4; z_5 z_6)
///     T8 = h_3(a ^ z_7; (b ^ z_7) z_8) ^ z_7
/// ```
///
/// Each `h_r(x; y z)` is one BLAKE3 compression truncated to 32 bytes:
///
/// - chaining value `x`, block `y || z`, block length 64;
/// - counter `r`, and the keyed, chunk-start and chunk-end flags.
///
/// A plain hash never sets the keyed flag, so no role meets the plain BLAKE3 node hash.
///
/// A record of `7k + 1` blocks chains `k` stages.
///
/// Each later stage takes the previous output as its `z_1`, then seven fresh blocks.
///
/// Records must be `32 + 224 k` bytes with `k >= 1`: 256, 480, 704, ...
///
/// Hashing any other length panics.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct T8Blake3;

impl CryptographicHasher<u8, [u8; OUT_LEN]> for T8Blake3 {
    const LANES: usize = crate::LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; OUT_LEN]
    where
        I: IntoIterator<Item = u8>,
    {
        // Feed the record through a stack buffer, in the same chunks the BLAKE3 hasher uses.
        const BUFLEN: usize = 512;
        let mut state = Streaming::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |chunk| state.update(chunk));
        state.finalize()
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; OUT_LEN]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        // A record given in one piece skips the streaming state.
        let mut pieces = input.into_iter();
        let Some(first) = pieces.next() else {
            return self.hash_slice(&[]);
        };
        let Some(second) = pieces.next() else {
            return self.hash_slice(first);
        };

        let mut state = Streaming::new();
        for piece in [first, second].into_iter().chain(pieces) {
            state.update(piece);
        }
        state.finalize()
    }

    fn hash_slice(&self, input: &[u8]) -> [u8; OUT_LEN] {
        let stages = stages(input.len()).expect("a T8 record is 32 + 224 k bytes for some k >= 1");
        debug_assert!(stages >= 1);

        // The record starts with z_1 of the first stage.
        let platform = Platform::detect();
        let (head, mut rest) = input
            .split_first_chunk::<BLOCK_BYTES>()
            .expect("checked length");
        let mut s = words(head);

        // Each stage consumes the next seven blocks, z_2 to z_8.
        while let Some((fresh, tail)) = rest.split_first_chunk::<STAGE_STRIDE>() {
            s = stage(platform, s, fresh);
            rest = tail;
        }
        bytes(&s)
    }

    /// Hash equal-length records laid end to end, one vector lane per record.
    ///
    /// # Panics
    ///
    /// Panics if the input length is not a whole multiple of the digest count.
    ///
    /// Panics if a record is not `32 + 224 k` bytes for some `k >= 1`.
    fn hash_many(&self, input: &[u8], out: &mut [[u8; OUT_LEN]]) {
        // No digests requested means there is nothing to read.
        if out.is_empty() {
            return;
        }

        // Every record has the same length, so the split is exact by contract.
        assert!(
            input.len().is_multiple_of(out.len()),
            "input length ({}) must be a whole multiple of the digest count ({})",
            input.len(),
            out.len()
        );
        let len = input.len() / out.len();
        assert!(
            stages(len).is_some(),
            "a T8 record is 32 + 224 k bytes for some k >= 1, not {len}"
        );
        crate::batch::t8::hash_many(input, len, out);
    }
}

/// One evaluation of T8 through the one-block compression of the `blake3` crate.
///
/// That routine also serves every block of a plain one-message BLAKE3 hash.
///
/// The seven fresh blocks sit at these byte offsets:
///
/// ```text
///     z_2 z_3   0..64
///     z_4      64..96
///     z_5 z_6  96..160
///     z_7     160..192
///     z_8     192..224
/// ```
#[inline]
fn stage(platform: Platform, z1: Words, fresh: &[u8; STAGE_STRIDE]) -> Words {
    let (flags, block_len) = (FLAGS as u8, 2 * BLOCK_BYTES as u8);
    let block = |at: usize| -> &[u8; 2 * BLOCK_BYTES] {
        fresh[at..][..2 * BLOCK_BYTES].try_into().expect("in range")
    };
    let word_block =
        |at: usize| -> Words { words(fresh[at..][..BLOCK_BYTES].try_into().expect("in range")) };

    // The two independent calls.
    let mut a = z1;
    platform.compress_in_place(&mut a, block(0), block_len, ROLES[0], flags);
    let mut b = word_block(64);
    platform.compress_in_place(&mut b, block(96), block_len, ROLES[1], flags);

    // The last call: chaining value a ^ z_7, block (b ^ z_7) || z_8.
    let z7 = word_block(160);
    let mut c: Words = core::array::from_fn(|i| a[i] ^ z7[i]);
    let mut last = [0u8; 2 * BLOCK_BYTES];
    last[..BLOCK_BYTES].copy_from_slice(&bytes(&core::array::from_fn(|i| b[i] ^ z7[i])));
    last[BLOCK_BYTES..].copy_from_slice(&fresh[192..]);
    platform.compress_in_place(&mut c, &last, block_len, ROLES[2], flags);

    core::array::from_fn(|i| c[i] ^ z7[i])
}

/// Read 32 bytes as eight little-endian words.
#[inline(always)]
fn words(block: &[u8; BLOCK_BYTES]) -> Words {
    let (chunks, _) = block.as_chunks::<4>();
    core::array::from_fn(|i| u32::from_le_bytes(chunks[i]))
}

/// Write eight words as 32 little-endian bytes.
#[inline(always)]
fn bytes(words: &Words) -> [u8; OUT_LEN] {
    let mut out = [0u8; OUT_LEN];
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *chunk = word.to_le_bytes();
    }
    out
}

/// A record fed a piece at a time.
///
/// It keeps at most one stage of pending bytes, whatever the record length.
struct Streaming {
    /// The detected one-block compression.
    platform: Platform,
    /// `z_1` of the next stage: the record's first block, then each stage output.
    s: Words,
    /// Bytes of the first block or of the next stage, not yet consumed.
    pending: [u8; STAGE_STRIDE],
    /// How many pending bytes are filled.
    filled: usize,
    /// Whether the first block has been read.
    started: bool,
    /// Stages completed so far.
    stages: usize,
}

impl Streaming {
    /// An empty record.
    fn new() -> Self {
        Self {
            platform: Platform::detect(),
            s: [0; 8],
            pending: [0; STAGE_STRIDE],
            filled: 0,
            started: false,
            stages: 0,
        }
    }

    /// Append bytes to the record.
    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // The first block goes to z_1, every later run of 224 bytes to one stage.
            let want = if self.started {
                STAGE_STRIDE
            } else {
                BLOCK_BYTES
            };
            let take = (want - self.filled).min(input.len());
            self.pending[self.filled..][..take].copy_from_slice(&input[..take]);
            self.filled += take;
            input = &input[take..];
            if self.filled < want {
                continue;
            }

            // A full piece: start the record, or run one stage.
            if self.started {
                self.s = stage(self.platform, self.s, &self.pending);
                self.stages += 1;
            } else {
                self.s = words(self.pending[..BLOCK_BYTES].try_into().expect("in range"));
                self.started = true;
            }
            self.filled = 0;
        }
    }

    /// The digest of the whole record.
    ///
    /// # Panics
    ///
    /// Panics unless the record is `32 + 224 k` bytes for some `k >= 1`.
    fn finalize(self) -> [u8; OUT_LEN] {
        assert!(
            self.stages >= 1 && self.filled == 0,
            "a T8 record is 32 + 224 k bytes for some k >= 1",
        );
        bytes(&self.s)
    }
}
