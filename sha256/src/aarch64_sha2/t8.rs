//! T8 leaves on the ARMv8 SHA-2 extension, as independent streams side by side.
//!
//! A chaining value and a 32-byte block have the same register form: two vectors of four words.
//!
//! ```text
//!     words 0..4 -> lo      words 4..8 -> hi
//! ```
//!
//! So the XORs of the last call run on the loaded vectors, with no trip through bytes.
//!
//! A batch runs four records per stream group, one call of a stage at a time.
//!
//! One record runs its two independent calls as two streams, so it waits on two calls, not three.

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, veorq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u8_u32,
    vreinterpretq_u32_u8, vrev32q_u8, vst1q_u8, vst1q_u32,
};

use p3_symmetric::CryptographicHasher;

use super::{ROUND_GROUPS, SCHEDULE_VECTORS, extend_schedule, round_group};
use crate::four_lane::LANES;
use crate::t8::{BLOCK_BYTES, STAGE_STRIDE};

/// Eight words, a chaining value or a 32-byte block, as a pair of vectors.
#[derive(Clone, Copy)]
pub(super) struct Pair {
    /// Words 0 to 3: `abcd` of a state, or the first four message words.
    pub(super) lo: uint32x4_t,
    /// Words 4 to 7: `efgh` of a state, or the next four message words.
    pub(super) hi: uint32x4_t,
}

impl Pair {
    /// Load 32 bytes as eight big-endian words.
    #[inline(always)]
    pub(super) fn load(bytes: &[u8; BLOCK_BYTES]) -> Self {
        // SAFETY: both 16-byte reads lie inside the 32-byte array.
        // The module compiles only with `neon`, which the loads and the byte reversal need.
        //
        // SHA-256 reads words big-endian and a vector load is little-endian, hence the reversal.
        unsafe {
            let p = bytes.as_ptr();
            Self {
                lo: vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p))),
                hi: vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(16)))),
            }
        }
    }

    /// Load eight words already in native order.
    #[inline(always)]
    fn from_words(words: &[u32; 8]) -> Self {
        // SAFETY: both four-word reads lie inside the eight-word array, and `neon` is enabled.
        unsafe {
            Self {
                lo: vld1q_u32(words.as_ptr()),
                hi: vld1q_u32(words.as_ptr().add(4)),
            }
        }
    }

    /// Store eight words in native order.
    #[inline(always)]
    fn to_words(self) -> [u32; 8] {
        let mut out = [0u32; 8];
        // SAFETY: both four-word writes lie inside the eight-word array, and `neon` is enabled.
        unsafe {
            vst1q_u32(out.as_mut_ptr(), self.lo);
            vst1q_u32(out.as_mut_ptr().add(4), self.hi);
        }
        out
    }

    /// Store as 32 big-endian bytes, the digest encoding.
    #[inline(always)]
    pub(super) fn store(self, out: &mut [u8; 32]) {
        // SAFETY: both 16-byte writes lie inside the 32-byte array, and `neon` is enabled.
        unsafe {
            let p = out.as_mut_ptr();
            vst1q_u8(p, vrev32q_u8(vreinterpretq_u8_u32(self.lo)));
            vst1q_u8(p.add(16), vrev32q_u8(vreinterpretq_u8_u32(self.hi)));
        }
    }

    /// Word-wise XOR.
    #[inline(always)]
    pub(super) fn xor(self, other: Self) -> Self {
        // SAFETY: `veorq_u32` is `neon`, which the module requires.
        unsafe {
            Self {
                lo: veorq_u32(self.lo, other.lo),
                hi: veorq_u32(self.hi, other.hi),
            }
        }
    }
}

/// One call `h(x; y z)` on each of `N` streams: `h[i] <- compress(h[i], y[i] || z[i])`.
///
/// The streams are independent, so the core overlaps their round chains.
#[inline(always)]
pub(super) fn compress<const N: usize>(h: &mut [Pair; N], y: [Pair; N], z: [Pair; N]) {
    let entry = *h;
    let mut schedule: [[uint32x4_t; SCHEDULE_VECTORS]; N] =
        core::array::from_fn(|i| [y[i].lo, y[i].hi, z[i].lo, z[i].hi]);

    // Rounds 0..16 read the block itself.
    for group in 0..SCHEDULE_VECTORS {
        for (state, window) in h.iter_mut().zip(&schedule) {
            round_group(&mut state.lo, &mut state.hi, window[group], group);
        }
    }

    // Rounds 16..64 extend the schedule in place: slot g % 4 holds W[4g-16 .. 4g-12].
    for group in SCHEDULE_VECTORS..ROUND_GROUPS {
        for (state, window) in h.iter_mut().zip(&mut schedule) {
            let next = extend_schedule(
                window[group % SCHEDULE_VECTORS],
                window[(group + 1) % SCHEDULE_VECTORS],
                window[(group + 2) % SCHEDULE_VECTORS],
                window[(group + 3) % SCHEDULE_VECTORS],
            );
            window[group % SCHEDULE_VECTORS] = next;
            round_group(&mut state.lo, &mut state.hi, next, group);
        }
    }

    // The feed-forward of the Davies-Meyer mode.
    for (state, saved) in h.iter_mut().zip(entry) {
        // SAFETY: `vaddq_u32` is `neon`, which the module requires.
        unsafe {
            state.lo = vaddq_u32(state.lo, saved.lo);
            state.hi = vaddq_u32(state.hi, saved.hi);
        }
    }
}

/// The 32-byte block `i` of a stage's seven fresh blocks, `i = 0` being `z_2`.
#[inline(always)]
fn block(fresh: &[u8; STAGE_STRIDE], i: usize) -> Pair {
    Pair::load(
        fresh[i * BLOCK_BYTES..][..BLOCK_BYTES]
            .try_into()
            .expect("in range"),
    )
}

/// One T8 stage on `N` records, stream `i` taking `z_1 = s[i]` and the fresh blocks `fresh[i]`.
///
/// ```text
///     a = h(z_1; z_2 z_3)             N streams
///     b = h(z_4; z_5 z_6)             N streams
///     s = h(a ^ z_7; (b ^ z_7) z_8) ^ z_7
/// ```
#[inline(always)]
fn stage_streams<const N: usize>(s: &mut [Pair; N], fresh: [&[u8; STAGE_STRIDE]; N]) {
    let at = |i: usize| -> [Pair; N] { core::array::from_fn(|r| block(fresh[r], i)) };

    compress(s, at(0), at(1));
    let mut b = at(2);
    compress(&mut b, at(3), at(4));

    let z7 = at(5);
    for (a, z) in s.iter_mut().zip(z7) {
        *a = a.xor(z);
    }
    compress(s, core::array::from_fn(|r| b[r].xor(z7[r])), at(6));
    for (c, z) in s.iter_mut().zip(z7) {
        *c = c.xor(z);
    }
}

/// One T8 stage on one record, with both independent calls in flight at once.
///
/// This is the stage every single-record path runs, and it matches [`stage_streams`] call for call.
#[inline]
pub(crate) fn stage(z1: [u32; 8], fresh: &[u8; STAGE_STRIDE]) -> [u32; 8] {
    // The two independent calls, as two streams.
    let mut ab = [Pair::from_words(&z1), block(fresh, 2)];
    compress(
        &mut ab,
        [block(fresh, 0), block(fresh, 3)],
        [block(fresh, 1), block(fresh, 4)],
    );
    let [a, b] = ab;

    // The last call, which waits on both.
    let z7 = block(fresh, 5);
    let mut c = [a.xor(z7)];
    compress(&mut c, [b.xor(z7)], [block(fresh, 6)]);
    c[0].xor(z7).to_words()
}

/// T8-hash equal-length records of `len` bytes laid end to end in `input`.
///
/// The caller guarantees `input.len() == len * out.len()` and `len = 32 + 224 stages`.
pub(crate) fn hash_many(input: &[u8], len: usize, stages: usize, out: &mut [[u8; 32]]) {
    // Whole groups of four records, one stream each.
    let (groups, rest) = out.as_chunks_mut::<LANES>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let lanes = core::array::from_fn(|r| &input[(index * LANES + r) * len..][..len]);
        group(lanes, stages, digests);
    }

    // The short final group, one record at a time.
    let first = groups.len() * LANES;
    for (record, digest) in input[first * len..].chunks_exact(len).zip(rest) {
        *digest = crate::T8Sha256.hash_slice(record);
    }
}

/// T8-hash four records of equal length, stage by stage.
#[inline(always)]
fn group(lanes: [&[u8]; LANES], stages: usize, out: &mut [[u8; 32]; LANES]) {
    let mut s: [Pair; LANES] =
        core::array::from_fn(|r| Pair::load(lanes[r][..BLOCK_BYTES].try_into().expect("in range")));
    for stage in 0..stages {
        let offset = BLOCK_BYTES + stage * STAGE_STRIDE;
        stage_streams(
            &mut s,
            core::array::from_fn(|r| {
                lanes[r][offset..][..STAGE_STRIDE]
                    .try_into()
                    .expect("in range")
            }),
        );
    }
    for (state, digest) in s.iter().zip(out) {
        state.store(digest);
    }
}
