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

/// Chaining values of every lane of every group.
type State = [[Vector; STATE_WORDS]; GROUPS];

/// Message blocks of every lane of every group.
type Block = [[Vector; BLOCK_WORDS]; GROUPS];

/// The inputs of the three calls of a stage, each in the place the rounds read it.
///
/// Every 32-byte block of the record is written once, straight into its slot:
///
/// ```text
///     z_1 -> h1      z_2 z_3 -> m1      z_4 -> h2      z_5 z_6 -> m2
///     z_7 -> z7      z_8 -> high half of m3
/// ```
struct Slots {
    /// Chaining value of the first call: `z_1`, then `a`, then the stage output `s`.
    h1: State,
    /// Block of the first call: `z_2 || z_3`.
    m1: Block,
    /// Chaining value of the second call: `z_4`, then `b`.
    h2: State,
    /// Block of the second call: `z_5 || z_6`.
    m2: Block,
    /// The block `z_7`, kept for both XORs of the last call.
    z7: State,
    /// Chaining value of the last call: `a ^ z_7`.
    h3: State,
    /// Block of the last call: `(b ^ z_7) || z_8`.
    m3: Block,
}

/// Where the low or the high half of a loaded 64-byte block goes.
#[derive(Clone, Copy)]
enum Slot {
    H1,
    M1Low,
    M1High,
    H2,
    M2Low,
    M2High,
    Z7,
    M3High,
    /// Past the end of the record: dropped.
    None,
}

impl Slots {
    /// Load the aligned 64-byte block at `offset` of every lane, and write its two halves into their slots.
    #[inline(always)]
    fn load(&mut self, lanes: &[&[u8]; LANES], offset: usize, low: Slot, high: Slot) {
        let q = load_block(&lanes.map(|lane| lane[offset..][..BLOCK_BYTES].try_into().unwrap()));
        for (g, q) in q.iter().enumerate() {
            let (lo, hi) = q.split_at(STATE_WORDS);
            self.put(low, g, lo);
            self.put(high, g, hi);
        }
    }

    /// Write eight words of group `g` into a slot.
    #[inline(always)]
    fn put(&mut self, slot: Slot, g: usize, half: &[Vector]) {
        let dst: &mut [Vector] = match slot {
            Slot::H1 => &mut self.h1[g],
            Slot::M1Low => &mut self.m1[g][..STATE_WORDS],
            Slot::M1High => &mut self.m1[g][STATE_WORDS..],
            Slot::H2 => &mut self.h2[g],
            Slot::M2Low => &mut self.m2[g][..STATE_WORDS],
            Slot::M2High => &mut self.m2[g][STATE_WORDS..],
            Slot::Z7 => &mut self.z7[g],
            Slot::M3High => &mut self.m3[g][STATE_WORDS..],
            Slot::None => return,
        };
        dst.copy_from_slice(half);
    }

    /// The last call of a stage, then its output into `h1` as the next stage's `z_1`.
    #[inline(always)]
    fn finish(&mut self) {
        for g in 0..GROUPS {
            for i in 0..STATE_WORDS {
                self.h3[g][i] = self.h1[g][i].xor(self.z7[g][i]);
                self.m3[g][i] = self.h2[g][i].xor(self.z7[g][i]);
            }
        }
        call(&mut self.h3, &self.m3, ROLES[2]);
        for g in 0..GROUPS {
            for i in 0..STATE_WORDS {
                self.h1[g][i] = self.h3[g][i].xor(self.z7[g][i]);
            }
        }
    }
}

/// One T8 call in the given role, out of line.
///
/// A stage's three calls then share one copy of the rounds, which keeps the stage loop in the core's decoded-instruction cache.
#[inline(never)]
fn call(h: &mut State, m: &Block, role: u64) {
    compress(h, m, role, false);
}

/// T8-hash the record of every lane, stage by stage.
///
/// ```text
///     a = h_1(z_1; z_2 z_3)
///     b = h_2(z_4; z_5 z_6)
///     s = h_3(a ^ z_7; (b ^ z_7) z_8) ^ z_7
/// ```
///
/// The record is read one aligned 64-byte block at a time, each transposed once, like a plain hash.
///
/// A stage's seven blocks start on alternate halves of a 64-byte block, so stages come in two shapes:
///
/// ```text
///     even stage, z_2 already loaded:    [z_3 z_4] [z_5 z_6] [z_7 z_8]                3 loads
///     odd stage:                         [z_2 z_3] [z_4 z_5] [z_6 z_7] [z_8 z_2']     4 loads
/// ```
///
/// `z_2'` opens the next stage, and lands in the first call's slot only after that call is done.
fn group(lanes: &[&[u8]; LANES], stages: usize, out: &mut [[u8; DIGEST_BYTES]; LANES]) {
    let zero = [[Vector::splat(0); STATE_WORDS]; GROUPS];
    let wide = [[Vector::splat(0); BLOCK_WORDS]; GROUPS];
    let mut slots = Slots {
        h1: zero,
        m1: wide,
        h2: zero,
        m2: wide,
        z7: zero,
        h3: zero,
        m3: wide,
    };
    let len = 32 + STAGE_STRIDE * stages;
    let at = |block: usize| block * 32;

    // The first 64 bytes: z_1 and the first stage's z_2.
    slots.load(lanes, 0, Slot::H1, Slot::M1Low);

    for stage in 0..stages {
        // Index of this stage's z_2 among the record's 32-byte blocks.
        let first = 1 + 7 * stage;

        if stage % 2 == 0 {
            // z_2 sits in the high half of the previous load.
            slots.load(lanes, at(first + 1), Slot::M1High, Slot::H2);
            call(&mut slots.h1, &slots.m1, ROLES[0]);
            slots.load(lanes, at(first + 3), Slot::M2Low, Slot::M2High);
            call(&mut slots.h2, &slots.m2, ROLES[1]);
            slots.load(lanes, at(first + 5), Slot::Z7, Slot::M3High);
        } else {
            slots.load(lanes, at(first), Slot::M1Low, Slot::M1High);
            call(&mut slots.h1, &slots.m1, ROLES[0]);
            slots.load(lanes, at(first + 2), Slot::H2, Slot::M2Low);
            slots.load(lanes, at(first + 4), Slot::M2High, Slot::Z7);
            call(&mut slots.h2, &slots.m2, ROLES[1]);

            // z_8, then the next stage's z_2; the last stage reads the block that ends the record.
            if stage + 1 < stages {
                slots.load(lanes, at(first + 6), Slot::M3High, Slot::M1Low);
            } else {
                slots.load(lanes, len - BLOCK_BYTES, Slot::None, Slot::M3High);
            }
        }
        slots.finish();
    }

    store_digests(&slots.h1, out);
}
