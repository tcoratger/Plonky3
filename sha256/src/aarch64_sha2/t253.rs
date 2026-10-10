//! T253 leaves on the ARMv8 SHA-2 extension, with the T8 kernel's streams and register form.
//!
//! A batch runs four records per stream group, one call of a stage at a time.
//!
//! One record runs its two independent calls as two streams, so it waits on two calls, not three.

use core::arch::aarch64::{vdupq_n_u8, vextq_u8, vld1q_u8, vreinterpretq_u32_u8, vrev32q_u8};

use p3_symmetric::CryptographicHasher;

use super::t8::{Pair, compress};
use crate::four_lane::LANES;
use crate::t253::{BLOCK_BYTES, ROLES, STAGE_STRIDE, TAGGED_BYTES};

/// A chaining value `role || u`, from the 31 bytes of `u` at `start` of the fresh bytes.
///
/// ```text
///     bytes  0..16 = role, u[0..15]      one ext of the role into u[0..16]
///     bytes 16..32 = u[15..31]           one load
/// ```
#[inline(always)]
fn tagged(role: u8, fresh: &[u8; STAGE_STRIDE], start: usize) -> Pair {
    let u: &[u8; TAGGED_BYTES] = fresh[start..][..TAGGED_BYTES].try_into().expect("in range");
    // SAFETY: both 16-byte reads lie inside the 31 bytes of `u`, at 0 and at 15.
    // The module compiles only with `neon`, which the loads, the ext and the byte reversal need.
    //
    // SHA-256 reads words big-endian and a vector load is little-endian, hence the reversal.
    unsafe {
        let p = u.as_ptr();
        let lo = vextq_u8::<15>(vdupq_n_u8(role), vld1q_u8(p));
        let hi = vld1q_u8(p.add(15));
        Pair {
            lo: vreinterpretq_u32_u8(vrev32q_u8(lo)),
            hi: vreinterpretq_u32_u8(vrev32q_u8(hi)),
        }
    }
}

/// The 32 bytes at `start` of the fresh bytes, as eight words.
#[inline(always)]
fn half(fresh: &[u8; STAGE_STRIDE], start: usize) -> Pair {
    Pair::load(fresh[start..][..BLOCK_BYTES].try_into().expect("in range"))
}

/// The last call of a stage, from `a`, `b` and the fresh bytes, masked with `z_7`.
///
/// ```text
///     s' = h(0x03 || u_8;  (a ^ z_7) || (b ^ z_7)) ^ z_7
/// ```
#[inline(always)]
fn finish<const N: usize>(
    a: [Pair; N],
    b: [Pair; N],
    fresh: [&[u8; STAGE_STRIDE]; N],
) -> [Pair; N] {
    let z7: [Pair; N] = core::array::from_fn(|r| half(fresh[r], 158));
    let mut c: [Pair; N] = core::array::from_fn(|r| tagged(ROLES[2], fresh[r], 190));
    compress(
        &mut c,
        core::array::from_fn(|r| a[r].xor(z7[r])),
        core::array::from_fn(|r| b[r].xor(z7[r])),
    );
    core::array::from_fn(|r| c[r].xor(z7[r]))
}

/// One T253 stage on one record, with both independent calls in flight at once.
///
/// ```text
///     a = h(0x01 || u_1;  s || y_3)       stream 0
///     b = h(0x02 || u_4;  m_2)            stream 1
/// ```
#[inline(always)]
fn stage_pair(s: Pair, fresh: &[u8; STAGE_STRIDE]) -> Pair {
    let mut ab = [tagged(ROLES[0], fresh, 0), tagged(ROLES[1], fresh, 63)];
    compress(
        &mut ab,
        [s, half(fresh, 94)],
        [half(fresh, 31), half(fresh, 126)],
    );
    let [a, b] = ab;
    finish([a], [b], [fresh])[0]
}

/// One T253 stage on one record, the chained value given and returned as bytes.
#[inline]
pub(crate) fn stage(s: &[u8; 32], fresh: &[u8; STAGE_STRIDE]) -> [u8; 32] {
    let mut out = [0u8; 32];
    stage_pair(Pair::load(s), fresh).store(&mut out);
    out
}

/// T253-hash one record of `32 + 221 k` bytes, the chained value kept in registers between stages.
///
/// The caller guarantees the record length.
#[inline]
pub(crate) fn hash_one(record: &[u8]) -> [u8; 32] {
    let (head, stages) = record.split_at(BLOCK_BYTES);
    let mut s = Pair::load(head.try_into().expect("in range"));
    for fresh in stages.as_chunks::<STAGE_STRIDE>().0 {
        s = stage_pair(s, fresh);
    }
    let mut out = [0u8; 32];
    s.store(&mut out);
    out
}

/// One T253 stage on four records, stream `r` taking the chained value `s[r]` and `fresh[r]`.
#[inline(always)]
fn stage_streams(s: &mut [Pair; LANES], fresh: [&[u8; STAGE_STRIDE]; LANES]) {
    let mut a = core::array::from_fn(|r| tagged(ROLES[0], fresh[r], 0));
    compress(&mut a, *s, core::array::from_fn(|r| half(fresh[r], 31)));
    let mut b = core::array::from_fn(|r| tagged(ROLES[1], fresh[r], 63));
    compress(
        &mut b,
        core::array::from_fn(|r| half(fresh[r], 94)),
        core::array::from_fn(|r| half(fresh[r], 126)),
    );
    *s = finish(a, b, fresh);
}

/// T253-hash equal-length records of `len` bytes laid end to end in `input`.
///
/// The caller guarantees `input.len() == len * out.len()` and `len = 32 + 221 stages`.
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
        *digest = crate::T253Sha256.hash_slice(record);
    }
}

/// T253-hash four records of equal length, stage by stage.
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
