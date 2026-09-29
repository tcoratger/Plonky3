use alloc::vec;
use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;
use proptest::prelude::*;

use super::{BLOCK_BYTES, STAGE_STRIDE, T8Blake3, stages};
use crate::batch::supported;
use crate::batch::t8::{FLAGS, ROLES};

/// Bytes in a digest.
const DIGEST_BYTES: usize = 32;

/// The BLAKE3 compression function, written from the specification with no shared code.
///
/// Returns the first eight words of the output, the truncation every role uses.
fn reference_compress(
    cv: [u32; 8],
    block: &[u8; 64],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 8] {
    const IV: [u32; 8] = [
        0x6A09_E667,
        0xBB67_AE85,
        0x3C6E_F372,
        0xA54F_F53A,
        0x510E_527F,
        0x9B05_688C,
        0x1F83_D9AB,
        0x5BE0_CD19,
    ];
    const PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

    let mut m: [u32; 16] =
        core::array::from_fn(|i| u32::from_le_bytes(block[4 * i..][..4].try_into().unwrap()));
    let mut v = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    let g = |v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32| {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(12);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(8);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(7);
    };
    for round in 0..7 {
        g(&mut v, 0, 4, 8, 12, m[0], m[1]);
        g(&mut v, 1, 5, 9, 13, m[2], m[3]);
        g(&mut v, 2, 6, 10, 14, m[4], m[5]);
        g(&mut v, 3, 7, 11, 15, m[6], m[7]);
        g(&mut v, 0, 5, 10, 15, m[8], m[9]);
        g(&mut v, 1, 6, 11, 12, m[10], m[11]);
        g(&mut v, 2, 7, 8, 13, m[12], m[13]);
        g(&mut v, 3, 4, 9, 14, m[14], m[15]);
        if round < 6 {
            m = core::array::from_fn(|i| m[PERM[i]]);
        }
    }
    core::array::from_fn(|i| v[i] ^ v[i + 8])
}

/// Eight little-endian words of a 32-byte block.
fn words(bytes: &[u8]) -> [u32; 8] {
    core::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..][..4].try_into().unwrap()))
}

/// The 32 little-endian bytes of eight words.
fn bytes(words: [u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in words.iter().enumerate() {
        out[4 * i..][..4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// One role `h_r(x; y z)`: chaining value `x`, block `y || z`.
fn h(role: u64, x: [u32; 8], y: [u32; 8], z: [u32; 8]) -> [u32; 8] {
    let mut block = [0u8; 64];
    block[..32].copy_from_slice(&bytes(y));
    block[32..].copy_from_slice(&bytes(z));
    reference_compress(x, &block, role, 64, FLAGS)
}

/// Lane-wise XOR of two blocks.
fn xor(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
    core::array::from_fn(|i| a[i] ^ b[i])
}

/// T8 on a record of `7k + 1` blocks, exactly as the construction states it.
///
/// ```text
///     s_1 = T8(x_1 .. x_8),   s_j = T8(s_{j-1}, x_{7j-5} .. x_{7j+1})
///     T8(z) = h_3(h_1(z_1 z_2 z_3) ^ z_7, h_2(z_4 z_5 z_6) ^ z_7, z_8) ^ z_7
/// ```
fn reference_t8(record: &[u8]) -> [u8; 32] {
    let blocks: Vec<[u32; 8]> = record
        .as_chunks::<32>()
        .0
        .iter()
        .map(|b| words(b))
        .collect();
    assert_eq!((blocks.len() - 1) % 7, 0);

    let mut s = blocks[0];
    for fresh in blocks[1..].as_chunks::<7>().0 {
        let z = [
            s, fresh[0], fresh[1], fresh[2], fresh[3], fresh[4], fresh[5], fresh[6],
        ];
        let a = h(ROLES[0], z[0], z[1], z[2]);
        let b = h(ROLES[1], z[3], z[4], z[5]);
        s = xor(h(ROLES[2], xor(a, z[6]), xor(b, z[6]), z[7]), z[6]);
    }
    bytes(s)
}

/// Deterministic, non-repeating bytes.
fn data(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// A record of `k` stages.
const fn record_len(k: usize) -> usize {
    BLOCK_BYTES + STAGE_STRIDE * k
}

#[test]
fn reference_compression_matches_blake3() {
    // Invariant: the oracle is BLAKE3's compression, pinned through the public BLAKE3 hashes.
    //
    // Fixture state: three hashes that each reduce to known compressions.
    //
    //     hash(m), |m| <= 64      one compression: IV, counter 0, START | END | ROOT
    //     keyed_hash(k, m), 64 B  one compression: key, counter 0, KEYED | START | END | ROOT
    //     hash(m), |m| = 2048     chunk 0, chunk 1 (counter 1), then one PARENT | ROOT
    const START: u32 = 1;
    const END: u32 = 2;
    const PARENT: u32 = 4;
    const ROOT: u32 = 8;
    const KEYED: u32 = 16;
    const IV: [u32; 8] = [
        0x6A09_E667,
        0xBB67_AE85,
        0x3C6E_F372,
        0xA54F_F53A,
        0x510E_527F,
        0x9B05_688C,
        0x1F83_D9AB,
        0x5BE0_CD19,
    ];

    // Short messages, zero-padded into one block.
    for len in [0, 1, 31, 32, 63, 64] {
        let m = data(len, len as u64);
        let mut block = [0u8; 64];
        block[..len].copy_from_slice(&m);
        let got = reference_compress(IV, &block, 0, len as u32, START | END | ROOT);
        assert_eq!(bytes(got), *blake3::hash(&m).as_bytes(), "len {len}");
    }

    // A keyed hash of one full block.
    let key: [u8; 32] = data(32, 7).try_into().unwrap();
    let m: [u8; 64] = data(64, 8).try_into().unwrap();
    let got = reference_compress(words(&key), &m, 0, 64, KEYED | START | END | ROOT);
    assert_eq!(bytes(got), *blake3::keyed_hash(&key, &m).as_bytes());

    // Two chunks: the second carries counter 1, then a parent joins them.
    let m = data(2048, 9);
    let chunk_cv = |chunk: &[u8], counter: u64| {
        let mut cv = IV;
        for (i, block) in chunk.as_chunks::<64>().0.iter().enumerate() {
            let flags = if i == 0 { START } else { 0 } | if i == 15 { END } else { 0 };
            cv = reference_compress(cv, block, counter, 64, flags);
        }
        cv
    };
    let mut parent = [0u8; 64];
    parent[..32].copy_from_slice(&bytes(chunk_cv(&m[..1024], 0)));
    parent[32..].copy_from_slice(&bytes(chunk_cv(&m[1024..], 1)));
    let got = reference_compress(IV, &parent, 0, 64, PARENT | ROOT);
    assert_eq!(bytes(got), *blake3::hash(&m).as_bytes());
}

#[test]
fn record_lengths() {
    // Invariant: a record is 32 + 224 k bytes with k >= 1.
    //
    //     256 -> 1 stage, 480 -> 2, 704 -> 3; everything else is rejected
    assert_eq!(stages(256), Some(1));
    assert_eq!(stages(480), Some(2));
    assert_eq!(stages(704), Some(3));
    for len in [0, 1, 31, 32, 224, 255, 257, 288, 479, 481, 512] {
        assert_eq!(stages(len), None, "len {len}");
    }
}

#[test]
fn known_answers() {
    // Invariant: the digests are fixed by the construction and its role encoding.
    //
    // Fixture state: records whose byte i is i mod 256, of one and two stages.
    //
    // Any change to the roles, the flags, the wiring or the byte order moves these digests.
    let one: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let two: Vec<u8> = (0..480).map(|i| i as u8).collect();
    assert_eq!(T8Blake3.hash_slice(&one), reference_t8(&one));
    assert_eq!(T8Blake3.hash_slice(&two), reference_t8(&two));
    assert_eq!(hex(&reference_t8(&one)), KAT_ONE_STAGE);
    assert_eq!(hex(&reference_t8(&two)), KAT_TWO_STAGES);
}

/// T8 of the bytes 0, 1, ..., 255.
const KAT_ONE_STAGE: &str = "41c82f5e7830e4f6b7d4582e3f408c364398ed22c7ab1f1fca95001948fc2407";

/// T8 of the bytes 0, 1, ..., 255, 0, ..., 223.
const KAT_TWO_STAGES: &str = "ed90a7ac7eb0cab68d23736282a1e0f4e0a8b8c04c68f61e9f0379a27b408b15";

/// Lowercase hexadecimal.
fn hex(bytes: &[u8]) -> alloc::string::String {
    use core::fmt::Write;
    let mut s = alloc::string::String::new();
    for b in bytes {
        write!(s, "{b:02x}").unwrap();
    }
    s
}

#[test]
fn roles_are_separated() {
    // Invariant: the three roles are three different functions of one 96-byte input.
    //
    // They also differ from the plain BLAKE3 compression the tree's node hash uses.
    let x = words(&data(32, 1));
    let y = words(&data(32, 2));
    let z = words(&data(32, 3));
    let outs = [
        h(ROLES[0], x, y, z),
        h(ROLES[1], x, y, z),
        h(ROLES[2], x, y, z),
    ];
    assert_ne!(outs[0], outs[1]);
    assert_ne!(outs[1], outs[2]);
    assert_ne!(outs[0], outs[2]);

    // A T8 digest is not the BLAKE3 hash of the same record.
    let record = data(256, 4);
    assert_ne!(
        T8Blake3.hash_slice(&record),
        *blake3::hash(&record).as_bytes()
    );
}

#[test]
fn single_record_paths_match_the_reference() {
    // Invariant: the slice, iterator and piecewise paths all compute the same T8.
    //
    // Fixture state: one to four stages, random bytes and the two constant extremes.
    for k in 1..=4 {
        let len = record_len(k);
        for record in [data(len, k as u64), vec![0u8; len], vec![0xFF; len]] {
            let expected = reference_t8(&record);
            assert_eq!(T8Blake3.hash_slice(&record), expected, "k {k}");
            assert_eq!(
                T8Blake3.hash_iter(record.iter().copied()),
                expected,
                "k {k}"
            );

            // Pieces of 1, 7, 31, 33 and 225 bytes cross every internal boundary.
            for piece in [1, 7, 31, 33, 225] {
                let pieces = record.chunks(piece);
                assert_eq!(
                    T8Blake3.hash_iter_slices(pieces),
                    expected,
                    "k {k} piece {piece}"
                );
            }
        }
    }
}

#[test]
fn every_backend_matches_the_reference() {
    // Invariant: each backend this CPU runs agrees with the reference on every batch shape.
    //
    // Fixture state: counts around every lane and group size, one to three stages.
    //
    //     widths 16 x 2 (AVX-512), 8 x 2 (AVX2), 4 (SSE2): counts on both sides of each
    let counts = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100,
    ];
    for kernel in supported() {
        for k in 1..=3 {
            let len = record_len(k);
            for count in counts {
                let input = data(len * count, (k * 1000 + count) as u64);
                let mut out = vec![[0u8; DIGEST_BYTES]; count];
                kernel.hash_many_t8(&input, len, &mut out);
                for (i, (record, digest)) in input.chunks_exact(len).zip(&out).enumerate() {
                    assert_eq!(
                        *digest,
                        reference_t8(record),
                        "{kernel:?} k {k} count {count} record {i}"
                    );
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "32 + 224 k")]
fn batches_reject_other_lengths() {
    // Two records of 257 bytes each: not a T8 record length.
    let mut out = [[0u8; DIGEST_BYTES]; 2];
    T8Blake3.hash_many(&[0u8; 514], &mut out);
}

#[test]
#[should_panic(expected = "32 + 224 k")]
fn single_records_reject_other_lengths() {
    // A 288-byte record ends 32 bytes into a second stage.
    let _ = T8Blake3.hash_iter([0u8; 288]);
}

proptest! {
    #[test]
    fn batches_match_one_record_at_a_time(
        count in 0usize..70,
        k in 1usize..=3,
        seed in any::<u64>(),
    ) {
        // Invariant: the batched digests are the single-record digests, in order.
        let len = record_len(k);
        let input = data(len * count, seed);
        let mut out = vec![[0u8; DIGEST_BYTES]; count];
        T8Blake3.hash_many(&input, &mut out);
        for (record, digest) in input.chunks_exact(len).zip(&out) {
            prop_assert_eq!(*digest, T8Blake3.hash_slice(record));
        }
    }
}
