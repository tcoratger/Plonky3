//! T8 leaves on the batched kernel, one vector lane per record.
//!
//! Equal-length records share every stage, role and flag.
//!
//! So each of the three calls of a stage advances a whole group, as a plain hash does.

use blake3::OUT_LEN;

use super::compress::{CHUNK_END, CHUNK_START, STATE_WORDS, compress};
use super::lanes::{Backend, Word, detect};
use super::{Block, Lanes, State};
use crate::t8::{BLOCK_BYTES, STAGE_STRIDE, stages};

/// Flag of the keyed hash mode, which a plain hash never sets.
const KEYED_HASH: u32 = 1 << 4;

/// Flags of the three roles: keyed hash, chunk start, chunk end, and no root.
pub(crate) const FLAGS: u32 = KEYED_HASH | CHUNK_START | CHUNK_END;

/// Counter words naming the three roles, so `h_1`, `h_2` and `h_3` are distinct functions.
pub(crate) const ROLES: [u64; 3] = [1, 2, 3];

/// Message bytes of each call: a full block.
const BLOCK_LEN: u32 = 2 * BLOCK_BYTES as u32;

/// T8-hash equal-length records of `len` bytes laid end to end in `input`.
///
/// The widest backend the running CPU supports does the work.
///
/// The caller guarantees `input.len() == len * out.len()` and a T8 record length.
pub(crate) fn hash_many(input: &[u8], len: usize, out: &mut [[u8; OUT_LEN]]) {
    detect().hash_many_t8(input, len, out);
}

/// T8-hash equal-length records with backend `V`, in groups of `G` registers of `W` lanes.
///
/// The caller guarantees `input.len() == len * out.len()` and a T8 record length.
///
/// # Safety
///
/// The running CPU has the target features of `V`.
pub(super) unsafe fn hash_many_with<V: Backend<W>, const W: usize, const G: usize>(
    input: &[u8],
    len: usize,
    out: &mut [[u8; OUT_LEN]],
) {
    debug_assert_eq!(input.len(), len * out.len());
    let stages = stages(len).expect("the caller checks the record length");

    // Full register groups first, then the single registers left over.
    let (registers, rest) = out.as_chunks_mut::<W>();
    let (groups, singles) = registers.as_chunks_mut::<G>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let first = index * W * G;
        let starts = core::array::from_fn(|g| core::array::from_fn(|l| (first + g * W + l) * len));
        let lanes = Lanes {
            batch: input,
            starts,
            last: (first + W * G - 1) * len,
        };
        // SAFETY: the caller runs this on a CPU with the features of `V`.
        unsafe { V::t8_group::<G>(&lanes, stages, digests) };
    }

    let mut first = groups.len() * W * G;
    for digests in singles {
        let lanes = Lanes {
            batch: input,
            starts: [core::array::from_fn(|l| (first + l) * len)],
            last: (first + W - 1) * len,
        };
        // SAFETY: the caller runs this on a CPU with the features of `V`.
        unsafe { V::t8_group::<1>(&lanes, stages, core::array::from_mut(digests)) };
        first += W;
    }

    // A short final register repeats its records to fill the spare lanes.
    //
    // Those lanes compute digests that are never written out.
    if !rest.is_empty() {
        let lanes = Lanes {
            batch: input,
            starts: [core::array::from_fn(|l| (first + l % rest.len()) * len)],
            last: (first + rest.len() - 1) * len,
        };
        let mut digests = [[[0u8; OUT_LEN]; W]];
        // SAFETY: the caller runs this on a CPU with the features of `V`.
        unsafe { V::t8_group::<1>(&lanes, stages, &mut digests) };
        rest.copy_from_slice(&digests[0][..rest.len()]);
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
///
/// The calls then leave `a` in `h1` and `b` in `h2`, and the wiring fills the rest of the last call.
struct Slots<V, const G: usize> {
    /// Chaining value of the first call: `z_1`, then `a`, then the stage output `s`.
    h1: State<V, G>,
    /// Block of the first call: `z_2 || z_3`.
    m1: Block<V, G>,
    /// Chaining value of the second call: `z_4`, then `b`.
    h2: State<V, G>,
    /// Block of the second call: `z_5 || z_6`.
    m2: Block<V, G>,
    /// The block `z_7`, kept for both XORs of the last call.
    z7: State<V, G>,
    /// Chaining value of the last call: `a ^ z_7`.
    h3: State<V, G>,
    /// Block of the last call: `(b ^ z_7) || z_8`.
    m3: Block<V, G>,
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

/// One T8 call with the given role counter.
///
/// It runs the backend's out-of-line step.
///
/// A stage's three calls then share one copy of the kernel, which keeps the stage loop in the core's decoded-instruction cache.
#[inline(always)]
fn call<V: Backend<W>, const W: usize, const G: usize>(
    h: &mut State<V, G>,
    m: &Block<V, G>,
    role: u64,
) {
    // SAFETY: the driver only reaches this backend on a CPU with its target features.
    unsafe { V::t8_compress(h, m, role) };
}

/// The body of the out-of-line step: one compression in the given role.
#[inline(always)]
pub(super) fn compress_role<V: Word, const G: usize>(
    h: &mut State<V, G>,
    m: &Block<V, G>,
    role: u64,
) {
    compress(h, m, role, BLOCK_LEN, FLAGS);
}

impl<V: Word, const G: usize> Slots<V, G> {
    /// Load the aligned 64-byte block at `offset` of every lane, and write its two halves into their slots.
    ///
    /// The slots are constants at every call site, so each write compiles to plain stores.
    #[inline(always)]
    fn load<const W: usize>(
        &mut self,
        lanes: &Lanes<'_, W, G>,
        offset: usize,
        low: Slot,
        high: Slot,
    ) where
        V: Backend<W>,
    {
        let q = lanes.load::<V>(offset);
        for (g, q) in q.iter().enumerate() {
            let (lo, hi) = q.split_at(STATE_WORDS);
            self.put(low, g, lo);
            self.put(high, g, hi);
        }
    }

    /// Write eight words of group `g` into a slot.
    #[inline(always)]
    fn put(&mut self, slot: Slot, g: usize, half: &[V]) {
        let dst: &mut [V] = match slot {
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
    ///
    /// ```text
    ///     h3 = a ^ z_7,   m3 = (b ^ z_7) || z_8,   s = h_3(h3; m3) ^ z_7
    /// ```
    #[inline(always)]
    fn finish<const W: usize>(&mut self)
    where
        V: Backend<W>,
    {
        for g in 0..G {
            for i in 0..STATE_WORDS {
                self.h3[g][i] = self.h1[g][i].xor(self.z7[g][i]);
                self.m3[g][i] = self.h2[g][i].xor(self.z7[g][i]);
            }
        }
        call(&mut self.h3, &self.m3, ROLES[2]);
        for g in 0..G {
            for i in 0..STATE_WORDS {
                self.h1[g][i] = self.h3[g][i].xor(self.z7[g][i]);
            }
        }
    }
}

/// T8-hash the record of every lane, stage by stage.
///
/// ```text
///     a = h_1(z_1; z_2 z_3)
///     b = h_2(z_4; z_5 z_6)
///     s = h_3(a ^ z_7; (b ^ z_7) z_8) ^ z_7
/// ```
///
/// From the second stage on, `z_1` is the previous `s` and the seven fresh blocks follow in the record.
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
#[inline(always)]
pub(super) fn group<V: Backend<W>, const W: usize, const G: usize>(
    lanes: &Lanes<'_, W, G>,
    stages: usize,
    out: &mut [[[u8; OUT_LEN]; W]; G],
) {
    let zero = [[V::splat(0); STATE_WORDS]; G];
    let wide = [[V::splat(0); 2 * STATE_WORDS]; G];
    let mut slots = Slots {
        h1: zero,
        m1: wide,
        h2: zero,
        m2: wide,
        z7: zero,
        h3: zero,
        m3: wide,
    };
    let len = BLOCK_BYTES + STAGE_STRIDE * stages;
    let at = |block: usize| block * BLOCK_BYTES;

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
                slots.load(lanes, len - 2 * BLOCK_BYTES, Slot::None, Slot::M3High);
            }
        }
        slots.finish::<W>();
    }

    for (s, out) in slots.h1.iter().zip(out) {
        V::store_digests(s, out);
    }
}
