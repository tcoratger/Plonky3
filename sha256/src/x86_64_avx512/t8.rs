//! T8 leaves on the 32-way AVX-512 kernel, one lane per record.
//!
//! Equal-length records share every stage, so each call of a stage advances all 32 lanes.

use core::arch::x86_64::*;

use p3_symmetric::CryptographicHasher;

use super::rounds::compress_blocks;
use super::{
    BLOCK_BYTES, BLOCK_WORDS, Block, GROUPS, LANES, Lanes, ONE_AT_A_TIME_BELOW, STATE_WORDS, State,
    load_block, store_digests,
};
use crate::t8::{STAGE_STRIDE, T8Sha256};

/// T8-hash equal-length records of `len` bytes laid end to end in `input`.
///
/// The caller guarantees `input.len() == len * out.len()` and a T8 record length.
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
            *digest = T8Sha256.hash_slice(record);
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

/// T8-hash the record of every lane, stage by stage.
///
/// A stage reads eight 32-byte blocks as four transposed message blocks:
///
/// ```text
///     block    q_0        q_1        q_2        q_3
///     words    z_1 z_2    z_3 z_4    z_5 z_6    z_7 z_8
///
///     a = h(z_1; z_2 z_3)
///     b = h(z_4; z_5 z_6)
///     s = h(a ^ z_7; (b ^ z_7) z_8) ^ z_7
/// ```
///
/// Stage `j` reads bytes `224 j .. 224 j + 256`.
///
/// From the second stage on, `z_1` is the previous `s`, and the block read in its place goes unused.
fn group(lanes: &Lanes<'_>, stages: usize, out: &mut [[u8; 32]; LANES]) {
    // Split a transposed block into its two 32-byte halves, and join two halves into a block.
    let low =
        |q: &[__m512i; BLOCK_WORDS]| -> [__m512i; STATE_WORDS] { core::array::from_fn(|i| q[i]) };
    let high = |q: &[__m512i; BLOCK_WORDS]| -> [__m512i; STATE_WORDS] {
        core::array::from_fn(|i| q[STATE_WORDS + i])
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
    // SAFETY: this module only compiles when the target enables AVX-512F.
    let xor = |x: &[__m512i; STATE_WORDS], y: &[__m512i; STATE_WORDS]| -> [__m512i; STATE_WORDS] {
        core::array::from_fn(|i| unsafe { _mm512_xor_si512(x[i], y[i]) })
    };

    let mut s: State = [[unsafe { _mm512_setzero_si512() }; STATE_WORDS]; GROUPS];
    for stage in 0..stages {
        let base = stage * STAGE_STRIDE;
        let q: [Block; 4] =
            core::array::from_fn(|k| load_block(&lanes.rows(base + k * BLOCK_BYTES)));

        // a = h(z_1; z_2 z_3), with z_1 read from the record only in the first stage.
        let mut a: State = core::array::from_fn(|g| if stage == 0 { low(&q[0][g]) } else { s[g] });
        let m: Block = core::array::from_fn(|g| join(&high(&q[0][g]), &low(&q[1][g])));
        compress_blocks(&mut a, &m);

        // b = h(z_4; z_5 z_6).
        let mut b: State = core::array::from_fn(|g| high(&q[1][g]));
        compress_blocks(&mut b, &q[2]);

        // s = h(a ^ z_7; (b ^ z_7) z_8) ^ z_7.
        let z7: State = core::array::from_fn(|g| low(&q[3][g]));
        let mut c: State = core::array::from_fn(|g| xor(&a[g], &z7[g]));
        let m: Block = core::array::from_fn(|g| join(&xor(&b[g], &z7[g]), &high(&q[3][g])));
        compress_blocks(&mut c, &m);
        s = core::array::from_fn(|g| xor(&c[g], &z7[g]));
    }

    store_digests(&s, out);
}
