//! T8 leaves on the batched kernel, one vector lane per record.
//!
//! Equal-length records share every stage and role, so each call advances a whole group.

use super::compress::{BLOCK_BYTES, BLOCK_WORDS, STATE_WORDS, compress};
use super::lanes::{GROUPS, LANES, Vector, Word, load_block, store_digests};
use crate::DIGEST_BYTES;
use crate::t8::{ROLES, STAGE_STRIDE};

/// One word per lane, for hashing a single record with the same rounds.
impl Word for u32 {
    #[inline(always)]
    fn splat(value: u32) -> Self {
        value
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        self & rhs
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    #[inline(always)]
    fn rotr_16(self) -> Self {
        self.rotate_right(16)
    }

    #[inline(always)]
    fn rotr_12(self) -> Self {
        self.rotate_right(12)
    }

    #[inline(always)]
    fn rotr_8(self) -> Self {
        self.rotate_right(8)
    }

    #[inline(always)]
    fn rotr_7(self) -> Self {
        self.rotate_right(7)
    }
}

/// One call `h_r(x; m)` on a single record: chaining value `x`, block `m`, counter `r`, not final.
#[inline]
pub(crate) fn compress_one(
    x: [u32; STATE_WORDS],
    m: &[u8; BLOCK_BYTES],
    role: u64,
) -> [u32; STATE_WORDS] {
    let (words, _) = m.as_chunks::<4>();
    let block = [core::array::from_fn(|w| u32::from_le_bytes(words[w]))];
    let mut state = [x];
    compress::<u32, 1>(&mut state, &block, role, false);
    state[0]
}

/// T8-hash equal-length records of `len` bytes laid end to end in `input`.
///
/// The caller guarantees `input.len() == len * out.len()` and `len = 32 + 224 stages`.
pub(crate) fn hash_many(input: &[u8], len: usize, stages: usize, out: &mut [[u8; DIGEST_BYTES]]) {
    // Whole groups first.
    let (groups, rest) = out.as_chunks_mut::<LANES>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let lanes = core::array::from_fn(|lane| &input[(index * LANES + lane) * len..][..len]);
        group(&lanes, stages, digests);
    }

    // A short final group repeats its records to fill the spare lanes.
    //
    // Those lanes compute digests that are never written out.
    if !rest.is_empty() {
        let first = groups.len() * LANES;
        let lanes = core::array::from_fn(|lane| &input[(first + lane % rest.len()) * len..][..len]);
        let mut digests = [[0u8; DIGEST_BYTES]; LANES];
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
///     a = h_1(z_1; z_2 z_3)
///     b = h_2(z_4; z_5 z_6)
///     s = h_3(a ^ z_7; (b ^ z_7) z_8) ^ z_7
/// ```
///
/// Stage `j` reads bytes `224 j .. 224 j + 256`.
///
/// From the second stage on, `z_1` is the previous `s`, and the block read in its place goes unused.
fn group(lanes: &[&[u8]; LANES], stages: usize, out: &mut [[u8; DIGEST_BYTES]; LANES]) {
    type State = [[Vector; STATE_WORDS]; GROUPS];
    type Block = [[Vector; BLOCK_WORDS]; GROUPS];

    // Split a transposed block into its two 32-byte halves, and join two halves into a block.
    let low =
        |q: &[Vector; BLOCK_WORDS]| -> [Vector; STATE_WORDS] { core::array::from_fn(|i| q[i]) };
    let high = |q: &[Vector; BLOCK_WORDS]| -> [Vector; STATE_WORDS] {
        core::array::from_fn(|i| q[STATE_WORDS + i])
    };
    let join = |x: &[Vector; STATE_WORDS], y: &[Vector; STATE_WORDS]| -> [Vector; BLOCK_WORDS] {
        core::array::from_fn(|i| {
            if i < STATE_WORDS {
                x[i]
            } else {
                y[i - STATE_WORDS]
            }
        })
    };
    let xor = |x: &[Vector; STATE_WORDS], y: &[Vector; STATE_WORDS]| -> [Vector; STATE_WORDS] {
        core::array::from_fn(|i| x[i].xor(y[i]))
    };

    let mut s: State = [[Vector::splat(0); STATE_WORDS]; GROUPS];
    for stage in 0..stages {
        let base = stage * STAGE_STRIDE;
        let q: [Block; 4] = core::array::from_fn(|k| {
            let offset = base + k * BLOCK_BYTES;
            load_block(&lanes.map(|lane| lane[offset..][..BLOCK_BYTES].try_into().unwrap()))
        });

        // a = h_1(z_1; z_2 z_3), with z_1 read from the record only in the first stage.
        let mut a: State = core::array::from_fn(|g| if stage == 0 { low(&q[0][g]) } else { s[g] });
        let m: Block = core::array::from_fn(|g| join(&high(&q[0][g]), &low(&q[1][g])));
        compress(&mut a, &m, ROLES[0], false);

        // b = h_2(z_4; z_5 z_6).
        let mut b: State = core::array::from_fn(|g| high(&q[1][g]));
        compress(&mut b, &q[2], ROLES[1], false);

        // s = h_3(a ^ z_7; (b ^ z_7) z_8) ^ z_7.
        let z7: State = core::array::from_fn(|g| low(&q[3][g]));
        let mut c: State = core::array::from_fn(|g| xor(&a[g], &z7[g]));
        let m: Block = core::array::from_fn(|g| join(&xor(&b[g], &z7[g]), &high(&q[3][g])));
        compress(&mut c, &m, ROLES[2], false);
        s = core::array::from_fn(|g| xor(&c[g], &z7[g]));
    }

    store_digests(&s, out);
}
