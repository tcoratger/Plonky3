//! T8 leaves on the batched kernel, one vector lane per record.
//!
//! Equal-length records share every stage, role and flag.
//!
//! So each of the three calls of a stage advances a whole group, as a plain hash does.

use blake3::OUT_LEN;

use super::compress::{CHUNK_END, CHUNK_START, STATE_WORDS, compress};
use super::lanes::{Backend, Word, detect};
use super::{Block, Lanes, State};
use crate::t8::{BLOCK_BYTES, stages};

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

impl<V: Word, const G: usize> Slots<V, G> {
    /// Write the 32-byte block of index `index` in the record, for group `g`, into its slot.
    ///
    /// Index 0 is `z_1`; after it, the blocks run `z_2 .. z_8` once per stage.
    #[inline(always)]
    fn put(&mut self, index: usize, g: usize, half: &[V]) {
        let dst: &mut [V] = if index == 0 {
            &mut self.h1[g]
        } else {
            match (index - 1) % 7 {
                0 => &mut self.m1[g][..STATE_WORDS],
                1 => &mut self.m1[g][STATE_WORDS..],
                2 => &mut self.h2[g],
                3 => &mut self.m2[g][..STATE_WORDS],
                4 => &mut self.m2[g][STATE_WORDS..],
                5 => &mut self.z7[g],
                _ => &mut self.m3[g][STATE_WORDS..],
            }
        };
        dst.copy_from_slice(half);
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
/// A block is read just before the call that needs it.
///
/// So the block that opens the next stage lands in the first call's slot only once that call is done.
#[inline(always)]
pub(super) fn group<V: Backend<W>, const W: usize, const G: usize>(
    lanes: &Lanes<'_, W, G>,
    stages: usize,
    out: &mut [[[u8; OUT_LEN]; W]; G],
) {
    let zero = [[V::splat(0); STATE_WORDS]; G];
    let mut slots = Slots {
        h1: zero,
        m1: [[V::splat(0); 2 * STATE_WORDS]; G],
        h2: zero,
        m2: [[V::splat(0); 2 * STATE_WORDS]; G],
        z7: zero,
        h3: zero,
        m3: [[V::splat(0); 2 * STATE_WORDS]; G],
    };

    // The record has 7 k + 1 blocks of 32 bytes; `loaded` of them sit in their slots.
    let blocks = 1 + 7 * stages;
    let len = BLOCK_BYTES * blocks;
    let mut loaded = 0;
    let mut fill = |slots: &mut Slots<V, G>, until: usize| {
        while loaded < until {
            // Two blocks per aligned 64-byte load.
            //
            // An odd count ends with one block: the high half of the load that ends the record.
            if loaded + 1 < blocks {
                let q = lanes.load::<V>(loaded * BLOCK_BYTES);
                for (g, q) in q.iter().enumerate() {
                    slots.put(loaded, g, &q[..STATE_WORDS]);
                    slots.put(loaded + 1, g, &q[STATE_WORDS..]);
                }
                loaded += 2;
            } else {
                let q = lanes.load::<V>(len - 2 * BLOCK_BYTES);
                for (g, q) in q.iter().enumerate() {
                    slots.put(loaded, g, &q[STATE_WORDS..]);
                }
                loaded += 1;
            }
        }
    };

    for stage in 0..stages {
        // Index of this stage's z_2 in the record.
        let first = 1 + 7 * stage;

        // a = h_1(z_1; z_2 z_3), left in the slot of z_1.
        fill(&mut slots, first + 2);
        compress(&mut slots.h1, &slots.m1, ROLES[0], BLOCK_LEN, FLAGS);

        // b = h_2(z_4; z_5 z_6), left in the slot of z_4.
        fill(&mut slots, first + 5);
        compress(&mut slots.h2, &slots.m2, ROLES[1], BLOCK_LEN, FLAGS);

        // The last call's inputs: chaining value a ^ z_7, block (b ^ z_7) || z_8.
        fill(&mut slots, first + 7);
        for g in 0..G {
            for i in 0..STATE_WORDS {
                slots.h3[g][i] = slots.h1[g][i].xor(slots.z7[g][i]);
                slots.m3[g][i] = slots.h2[g][i].xor(slots.z7[g][i]);
            }
        }
        compress(&mut slots.h3, &slots.m3, ROLES[2], BLOCK_LEN, FLAGS);

        // s = h_3(...) ^ z_7, which is also the next stage's z_1.
        for g in 0..G {
            for i in 0..STATE_WORDS {
                slots.h1[g][i] = slots.h3[g][i].xor(slots.z7[g][i]);
            }
        }
    }

    for (s, out) in slots.h1.iter().zip(out) {
        V::store_digests(s, out);
    }
}
