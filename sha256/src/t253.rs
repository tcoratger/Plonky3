//! T253 leaves: T8 on the SHA-256 compression function, with its three calls told apart.
//!
//! One SHA-256 compression reads exactly 96 bytes, so T8 on it has no room left for a role tag.
//!
//! T253 gives up one byte of each call's chaining value for that tag, so a record's first stage is 253 bytes.

#[cfg(test)]
mod tests;

use p3_symmetric::CryptographicHasher;

/// Bytes in a digest, and in the stage output that chains into the next stage.
pub const BLOCK_BYTES: usize = 32;

/// Payload bytes beside the role tag in a chaining value.
pub const TAGGED_BYTES: usize = BLOCK_BYTES - 1;

/// Fresh bytes per stage: three tagged 31-byte values, `y_3`, `m_2` and `z_7`.
///
/// ```text
///     u_1 (31)   y_3 (32)   u_4 (31)   m_2 (64)   z_7 (32)   u_8 (31)    221 bytes
/// ```
pub const STAGE_STRIDE: usize = 3 * TAGGED_BYTES + BLOCK_BYTES + 2 * BLOCK_BYTES + BLOCK_BYTES;

/// The role tags, in the top byte of each call's chaining value.
///
/// The tree's node hash always starts from the SHA-256 initial value, whose top byte is 0x6a.
///
/// So no role meets it, and the three roles are three distinct functions.
pub(crate) const ROLES: [u8; 3] = [0x01, 0x02, 0x03];

/// Eight big-endian words: a chaining value, a digest, or a 32-byte block.
type Words = [u32; 8];

/// Stages in a record of `len` bytes.
///
/// Returns `None` unless `len = 32 + 221 k` with `k >= 1`.
#[must_use]
pub const fn stages(len: usize) -> Option<usize> {
    match len.checked_sub(BLOCK_BYTES) {
        Some(rest) if rest >= STAGE_STRIDE && rest.is_multiple_of(STAGE_STRIDE) => {
            Some(rest / STAGE_STRIDE)
        }
        _ => None,
    }
}

/// T8 on the SHA-256 compression function, with a role tag in each call's chaining value.
///
/// A record is a 32-byte value `s_0`, then `k` stages of 221 bytes.
///
/// Each stage runs three calls, `h(cv; block)` being one SHA-256 compression without padding:
///
/// ```text
///     a  = h(0x01 || u_1;  s || y_3)
///     b  = h(0x02 || u_4;  m_2)
///     s' = h(0x03 || u_8;  (a ^ z_7) || (b ^ z_7)) ^ z_7
/// ```
///
/// This is the construction's generalized form with its XORs unchanged, over role-separated calls.
///
/// The chained value `s` enters the first call only, as `z_1` does in T8.
///
/// Records must be `32 + 221 k` bytes with `k >= 1`: 253, 474, 695, ...
///
/// Hashing any other length panics.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct T253Sha256;

impl CryptographicHasher<u8, [u8; 32]> for T253Sha256 {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    const LANES: usize = crate::x86_64_avx512::LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        // Gather the record, a stage at a time, in the same chunks the SHA-256 hasher uses.
        const BUFLEN: usize = 512;
        let mut record = Streaming::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |chunk| record.update(chunk));
        record.finalize()
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

        let mut record = Streaming::new();
        for piece in [first, second].into_iter().chain(pieces) {
            record.update(piece);
        }
        record.finalize()
    }

    fn hash_slice(&self, input: &[u8]) -> [u8; 32] {
        stages(input.len()).expect("a T253 record is 32 + 221 k bytes for some k >= 1");

        // The record opens with s_0, then one stage per 221 bytes.
        let (head, mut rest) = input
            .split_first_chunk::<BLOCK_BYTES>()
            .expect("checked length");
        let mut s = *head;
        while let Some((fresh, tail)) = rest.split_first_chunk::<STAGE_STRIDE>() {
            s = stage(&s, fresh);
            rest = tail;
        }
        s
    }

    /// Hash equal-length records laid end to end.
    ///
    /// # Panics
    ///
    /// Panics if the input length is not a whole multiple of the digest count.
    ///
    /// Panics if a record is not `32 + 221 k` bytes for some `k >= 1`.
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
        let stages = stages(len).expect("a T253 record is 32 + 221 k bytes for some k >= 1");

        // The AVX-512 kernel when the build has it, one record at a time otherwise.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        crate::x86_64_avx512::t253::hash_many(input, len, stages, out);
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )))]
        {
            let _ = stages;
            for (record, digest) in input.chunks_exact(len).zip(out) {
                *digest = self.hash_slice(record);
            }
        }
    }
}

/// One stage, given the chained value `s` and the stage's 221 fresh bytes.
///
/// The fresh bytes sit at these offsets:
///
/// ```text
///     u_1   0..31     y_3  31..63     u_4  63..94
///     m_2  94..158    z_7 158..190    u_8 190..221
/// ```
#[inline]
fn stage(s: &[u8; 32], fresh: &[u8; STAGE_STRIDE]) -> [u8; 32] {
    let at = |start: usize, len: usize| &fresh[start..start + len];

    // A chaining value: the role tag, then 31 payload bytes.
    let tagged = |role: u8, payload: &[u8]| -> Words {
        let mut cv = [0u8; 32];
        cv[0] = role;
        cv[1..].copy_from_slice(payload);
        words(&cv)
    };

    // The first call's block: the chained value, then y_3.
    let mut m1 = [0u8; 64];
    m1[..32].copy_from_slice(s);
    m1[32..].copy_from_slice(at(31, 32));
    let m2: &[u8; 64] = at(94, 64).try_into().expect("in range");

    // The two independent calls, as two SHA-NI streams when the CPU has them.
    let (cv1, cv2) = (tagged(ROLES[0], at(0, 31)), tagged(ROLES[1], at(63, 31)));
    let (a, b) = pair((&cv1, &m1), (&cv2, m2));

    // The last call: block (a ^ z_7) || (b ^ z_7), then the output masked with z_7 again.
    let z7 = words(at(158, 32).try_into().expect("in range"));
    let mut m3 = [0u8; 64];
    m3[..32].copy_from_slice(&bytes(&core::array::from_fn(|i| a[i] ^ z7[i])));
    m3[32..].copy_from_slice(&bytes(&core::array::from_fn(|i| b[i] ^ z7[i])));
    let c = compress(tagged(ROLES[2], at(190, 31)), &m3);

    bytes(&core::array::from_fn(|i| c[i] ^ z7[i]))
}

/// The two independent calls of a stage.
#[inline]
fn pair(a: (&Words, &[u8; 64]), b: (&Words, &[u8; 64])) -> (Words, Words) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if crate::x86_64_sha_ni_pair::supported() {
        // SAFETY: the CPU check above found SHA-NI and SSE4.1.
        return unsafe { crate::x86_64_sha_ni_pair::compress_pair(a, b) };
    }
    (compress(*a.0, a.1), compress(*b.0, b.1))
}

/// One SHA-256 compression without padding, through `sha2`.
#[inline]
fn compress(mut state: Words, block: &[u8; 64]) -> Words {
    sha2::block_api::compress256(&mut state, core::slice::from_ref(block));
    state
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
    /// The chained value: `s_0`, then each stage output.
    s: [u8; 32],
    /// Bytes of `s_0` or of the next stage, not yet consumed.
    pending: [u8; STAGE_STRIDE],
    /// How many pending bytes are filled.
    filled: usize,
    /// Whether `s_0` has been read.
    started: bool,
    /// Stages completed so far.
    stages: usize,
}

impl Streaming {
    /// An empty record.
    const fn new() -> Self {
        Self {
            s: [0; 32],
            pending: [0; STAGE_STRIDE],
            filled: 0,
            started: false,
            stages: 0,
        }
    }

    /// Append bytes to the record.
    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // The first 32 bytes are s_0, every later run of 221 bytes one stage.
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
                self.s = stage(&self.s, &self.pending);
                self.stages += 1;
            } else {
                self.s.copy_from_slice(&self.pending[..BLOCK_BYTES]);
                self.started = true;
            }
            self.filled = 0;
        }
    }

    /// The digest of the whole record.
    ///
    /// # Panics
    ///
    /// Panics unless the record is `32 + 221 k` bytes for some `k >= 1`.
    fn finalize(self) -> [u8; 32] {
        assert!(
            self.stages >= 1 && self.filled == 0,
            "a T253 record is 32 + 221 k bytes for some k >= 1",
        );
        self.s
    }
}
