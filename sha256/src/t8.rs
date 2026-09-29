//! T8 leaves: records of eight 32-byte blocks hashed with three SHA-256 compressions.
//!
//! A plain SHA-256 hash of the same 256 bytes takes five: four blocks and one of padding.

#[cfg(test)]
mod tests;

use p3_symmetric::CryptographicHasher;

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

/// The T8 leaf hash of Dodis, Khovratovich, Mouha and Nandi, on the SHA-256 compression function.
///
/// One stage hashes eight blocks with three calls:
///
/// ```text
///     a = h(z_1; z_2 z_3)
///     b = h(z_4; z_5 z_6)
///     T8 = h(a ^ z_7; (b ^ z_7) z_8) ^ z_7
/// ```
///
/// Each `h(x; y z)` is one SHA-256 compression: chaining value `x`, block `y || z`, no padding.
///
/// The compression takes exactly 96 bytes, so no bit is left to tell the three calls apart.
///
/// All three calls are therefore the same function.
///
/// The construction's security analysis assumes three independent functions, so it does not cover this instantiation.
///
/// A record of `7k + 1` blocks chains `k` stages, each later stage taking the previous output as its `z_1`.
///
/// Records must be `32 + 224 k` bytes with `k >= 1`, and hashing any other length panics.
///
/// Batches run on the AVX-512 kernel, or four streams of the ARM SHA-2 extension, when the build enables one.
///
/// Any other build hashes one record at a time.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct T8Sha256;

impl CryptographicHasher<u8, [u8; 32]> for T8Sha256 {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    const LANES: usize = crate::x86_64_avx512::LANES;
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "sha2"
    ))]
    const LANES: usize = crate::four_lane::LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        // Feed the record through a stack buffer, in the same chunks the SHA-256 hasher uses.
        const BUFLEN: usize = 512;
        let mut state = Streaming::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |chunk| state.update(chunk));
        state.finalize()
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
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

    fn hash_slice(&self, input: &[u8]) -> [u8; 32] {
        stages(input.len()).expect("a T8 record is 32 + 224 k bytes for some k >= 1");

        // The record starts with z_1 of the first stage.
        let (head, mut rest) = input
            .split_first_chunk::<BLOCK_BYTES>()
            .expect("checked length");
        let mut s = words(head);

        // Each stage consumes the next seven blocks, z_2 to z_8.
        while let Some((fresh, tail)) = rest.split_first_chunk::<STAGE_STRIDE>() {
            s = stage(s, fresh);
            rest = tail;
        }
        bytes(&s)
    }

    /// Hash equal-length records laid end to end.
    ///
    /// # Panics
    ///
    /// Panics if the input length is not a whole multiple of the digest count.
    ///
    /// Panics if a record is not `32 + 224 k` bytes for some `k >= 1`.
    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
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
        let stages = stages(len).expect("a T8 record is 32 + 224 k bytes for some k >= 1");

        // The AVX-512 or the ARM SHA-2 kernel when the build has one, one record at a time otherwise.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        crate::x86_64_avx512::t8::hash_many(input, len, stages, out);
        #[cfg(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "sha2"
        ))]
        crate::aarch64_sha2::t8::hash_many(input, len, stages, out);
        #[cfg(not(any(
            all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            ),
            all(
                target_arch = "aarch64",
                target_feature = "neon",
                target_feature = "sha2"
            )
        )))]
        {
            let _ = stages;
            for (record, digest) in input.chunks_exact(len).zip(out) {
                *digest = self.hash_slice(record);
            }
        }
    }
}

/// One evaluation of T8, given `z_1` as words and `z_2 .. z_8` as bytes.
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
///
/// With the ARM SHA-2 extension, the two independent calls run side by side as two streams.
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha2"
))]
#[inline]
fn stage(z1: Words, fresh: &[u8; STAGE_STRIDE]) -> Words {
    crate::aarch64_sha2::t8::stage(z1, fresh)
}

/// One evaluation of T8 through `sha2`, one call at a time.
#[cfg(not(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha2"
)))]
#[inline]
fn stage(z1: Words, fresh: &[u8; STAGE_STRIDE]) -> Words {
    let block = |at: usize| -> &[u8; 2 * BLOCK_BYTES] {
        fresh[at..][..2 * BLOCK_BYTES].try_into().expect("in range")
    };
    let word_block =
        |at: usize| -> Words { words(fresh[at..][..BLOCK_BYTES].try_into().expect("in range")) };
    let compress = |mut state: Words, block: &[u8; 2 * BLOCK_BYTES]| {
        sha2::block_api::compress256(&mut state, core::slice::from_ref(block));
        state
    };

    // The two independent calls.
    let a = compress(z1, block(0));
    let b = compress(word_block(64), block(96));

    // The last call: chaining value a ^ z_7, block (b ^ z_7) || z_8.
    let z7 = word_block(160);
    let mut last = [0u8; 2 * BLOCK_BYTES];
    last[..BLOCK_BYTES].copy_from_slice(&bytes(&core::array::from_fn(|i| b[i] ^ z7[i])));
    last[BLOCK_BYTES..].copy_from_slice(&fresh[192..]);
    let c = compress(core::array::from_fn(|i| a[i] ^ z7[i]), &last);

    core::array::from_fn(|i| c[i] ^ z7[i])
}

/// Read 32 bytes as eight big-endian words, the order SHA-256 uses.
#[inline(always)]
fn words(block: &[u8; BLOCK_BYTES]) -> Words {
    let (chunks, _) = block.as_chunks::<4>();
    core::array::from_fn(|i| u32::from_be_bytes(chunks[i]))
}

/// Write eight words as 32 big-endian bytes.
#[inline(always)]
fn bytes(words: &Words) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *chunk = word.to_be_bytes();
    }
    out
}

/// A record fed a piece at a time.
///
/// It keeps at most one stage of pending bytes, whatever the record length.
struct Streaming {
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
    const fn new() -> Self {
        Self {
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
                self.s = stage(self.s, &self.pending);
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
    fn finalize(self) -> [u8; 32] {
        assert!(
            self.stages >= 1 && self.filled == 0,
            "a T8 record is 32 + 224 k bytes for some k >= 1",
        );
        bytes(&self.s)
    }
}
