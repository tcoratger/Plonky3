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

/// The inputs of the three calls of a stage, each in the place the kernel reads it.
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
    fn load(&mut self, lanes: &Lanes<'_>, offset: usize, low: Slot, high: Slot) {
        let q = load_block(&lanes.rows(offset));
        for (g, q) in q.iter().enumerate() {
            let (lo, hi) = q.split_at(STATE_WORDS);
            self.put(low, g, lo);
            self.put(high, g, hi);
        }
    }

    /// Write eight words of group `g` into a slot.
    #[inline(always)]
    fn put(&mut self, slot: Slot, g: usize, half: &[__m512i]) {
        let dst: &mut [__m512i] = match slot {
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
        // SAFETY: this module only compiles when the target enables AVX-512F.
        let xor = |x: __m512i, y: __m512i| unsafe { _mm512_xor_si512(x, y) };
        for g in 0..GROUPS {
            for i in 0..STATE_WORDS {
                self.h3[g][i] = xor(self.h1[g][i], self.z7[g][i]);
                self.m3[g][i] = xor(self.h2[g][i], self.z7[g][i]);
            }
        }
        call(&mut self.h3, &self.m3);
        for g in 0..GROUPS {
            for i in 0..STATE_WORDS {
                self.h1[g][i] = xor(self.h3[g][i], self.z7[g][i]);
            }
        }
    }
}

/// One T8 call, out of line.
///
/// A stage's three calls then share one copy of the kernel, which keeps the stage loop in the core's decoded-instruction cache.
#[inline(never)]
fn call(h: &mut State, m: &Block) {
    compress_blocks(h, m);
}

/// T8-hash the record of every lane, stage by stage.
///
/// ```text
///     a = h(z_1; z_2 z_3)
///     b = h(z_4; z_5 z_6)
///     s = h(a ^ z_7; (b ^ z_7) z_8) ^ z_7
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
fn group(lanes: &Lanes<'_>, stages: usize, out: &mut [[u8; 32]; LANES]) {
    // SAFETY: this module only compiles when the target enables AVX-512F.
    let zero = unsafe { _mm512_setzero_si512() };
    let mut slots = Slots {
        h1: [[zero; STATE_WORDS]; GROUPS],
        m1: [[zero; BLOCK_WORDS]; GROUPS],
        h2: [[zero; STATE_WORDS]; GROUPS],
        m2: [[zero; BLOCK_WORDS]; GROUPS],
        z7: [[zero; STATE_WORDS]; GROUPS],
        h3: [[zero; STATE_WORDS]; GROUPS],
        m3: [[zero; BLOCK_WORDS]; GROUPS],
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
            call(&mut slots.h1, &slots.m1);
            slots.load(lanes, at(first + 3), Slot::M2Low, Slot::M2High);
            call(&mut slots.h2, &slots.m2);
            slots.load(lanes, at(first + 5), Slot::Z7, Slot::M3High);
        } else {
            slots.load(lanes, at(first), Slot::M1Low, Slot::M1High);
            call(&mut slots.h1, &slots.m1);
            slots.load(lanes, at(first + 2), Slot::H2, Slot::M2Low);
            slots.load(lanes, at(first + 4), Slot::M2High, Slot::Z7);
            call(&mut slots.h2, &slots.m2);

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
