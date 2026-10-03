//! T253 leaves on the 32-way AVX-512 kernel, one lane per record.

use core::arch::x86_64::*;

use p3_symmetric::CryptographicHasher;

use super::rounds::compress_blocks;
use super::{
    BLOCK_WORDS, Block, LANES, Lanes, ONE_AT_A_TIME_BELOW, STATE_WORDS, State, load_block,
    store_digests,
};
use crate::t253::{ROLES, STAGE_STRIDE, T253Sha256};

/// T253-hash equal-length records of `len` bytes laid end to end in `input`.
///
/// The caller guarantees `input.len() == len * out.len()` and a T253 record length.
pub(crate) fn hash_many(input: &[u8], len: usize, stages: usize, out: &mut [[u8; 32]]) {
    // Whole groups of 32 records.
    let (groups, rest) = out.as_chunks_mut::<LANES>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let lanes = Lanes {
            input,
            len,
            first: index * LANES,
            count: LANES,
        };
        group(&lanes, stages, digests);
    }

    // The short final group, split as the plain hash splits it.
    let first = groups.len() * LANES;
    if rest.len() < ONE_AT_A_TIME_BELOW {
        for (record, digest) in input[first * len..].chunks_exact(len).zip(&mut *rest) {
            *digest = T253Sha256.hash_slice(record);
        }
    } else if !rest.is_empty() {
        // The spare lanes repeat the last record, and their digests are never written out.
        let lanes = Lanes {
            input,
            len,
            first,
            count: rest.len(),
        };
        let mut digests = [[0u8; 32]; LANES];
        group(&lanes, stages, &mut digests);
        rest.copy_from_slice(&digests[..rest.len()]);
    }
}

/// One call, out of line, so a stage's three calls share one copy of the kernel.
#[inline(never)]
fn call(h: &mut State, m: &Block) {
    compress_blocks(h, m);
}

/// T253-hash the record of every lane, stage by stage.
///
/// A stage at byte `o` reads four 64-byte rows, each split into two 32-byte halves:
///
/// ```text
///     row at o - 1     tag + u_1   | y_3
///     row at o + 62    tag + u_4   | m_2, first half
///     row at o + 126   m_2, second half | z_7
///     row at o + 189   tag + u_8   | (unused)
/// ```
///
/// A chaining-value half starts one byte early, on the byte before its 31-byte field.
///
/// That byte is replaced by the call's role tag.
///
/// The last stage reads its final row 32 bytes earlier, ending exactly at the record's end.
fn group(lanes: &Lanes<'_>, stages: usize, out: &mut [[u8; 32]; LANES]) {
    // SAFETY (every intrinsic below): this module only compiles when the target enables AVX-512F.
    let splat = |w: u32| unsafe { _mm512_set1_epi32(w as i32) };
    let xor = |x: __m512i, y: __m512i| unsafe { _mm512_xor_si512(x, y) };
    let keep_low = splat(0x00ff_ffff);

    // Split a loaded row into its halves, and put a role tag over the top byte of a chaining value.
    let low =
        |q: &[__m512i; BLOCK_WORDS]| -> [__m512i; STATE_WORDS] { core::array::from_fn(|i| q[i]) };
    let high = |q: &[__m512i; BLOCK_WORDS]| -> [__m512i; STATE_WORDS] {
        core::array::from_fn(|i| q[STATE_WORDS + i])
    };
    let tag = |mut cv: [__m512i; STATE_WORDS], role: u8| {
        cv[0] = unsafe {
            _mm512_or_si512(
                _mm512_and_si512(cv[0], keep_low),
                splat(u32::from(role) << 24),
            )
        };
        cv
    };
    let join = |x: &[__m512i; STATE_WORDS], y: &[__m512i; STATE_WORDS]| -> [__m512i; BLOCK_WORDS] {
        core::array::from_fn(|i| {
            if i < STATE_WORDS {
                x[i]
            } else {
                y[i - STATE_WORDS]
            }
        })
    };
    let row = |offset: usize| load_block(&lanes.rows(offset));

    // s_0: the record's first 32 bytes.
    let first = row(0);
    let mut s: State = core::array::from_fn(|g| low(&first[g]));

    for stage in 0..stages {
        let o = crate::t253::BLOCK_BYTES + STAGE_STRIDE * stage;
        let (r1, r2, r3) = (row(o - 1), row(o + 62), row(o + 126));

        // a = h(0x01 || u_1; s || y_3).
        let mut a: State = core::array::from_fn(|g| tag(low(&r1[g]), ROLES[0]));
        let m1: Block = core::array::from_fn(|g| join(&s[g], &high(&r1[g])));
        call(&mut a, &m1);

        // b = h(0x02 || u_4; m_2).
        let mut b: State = core::array::from_fn(|g| tag(low(&r2[g]), ROLES[1]));
        let m2: Block = core::array::from_fn(|g| join(&high(&r2[g]), &low(&r3[g])));
        call(&mut b, &m2);

        // s' = h(0x03 || u_8; (a ^ z_7) || (b ^ z_7)) ^ z_7.
        let z7: State = core::array::from_fn(|g| high(&r3[g]));
        let mut c: State = if stage + 1 < stages {
            let r4 = row(o + 189);
            core::array::from_fn(|g| tag(low(&r4[g]), ROLES[2]))
        } else {
            let r4 = row(o + 157);
            core::array::from_fn(|g| tag(high(&r4[g]), ROLES[2]))
        };
        let m3: Block = core::array::from_fn(|g| {
            join(
                &core::array::from_fn(|i| xor(a[g][i], z7[g][i])),
                &core::array::from_fn(|i| xor(b[g][i], z7[g][i])),
            )
        });
        call(&mut c, &m3);
        s = core::array::from_fn(|g| core::array::from_fn(|i| xor(c[g][i], z7[g][i])));
    }

    store_digests(&s, out);
}
