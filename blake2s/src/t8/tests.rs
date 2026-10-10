use alloc::vec;
use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;
use proptest::prelude::*;

use super::{BLOCK_BYTES, ROLES, STAGE_STRIDE, T8Blake2s, stages};

/// The BLAKE2s compression function of RFC 7693 section 3.2, written with no shared code.
fn reference_compress(h: [u32; 8], block: &[u8; 64], t: u64, last: bool) -> [u32; 8] {
    const IV: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const SIGMA: [[usize; 16]; 10] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    ];
    let m: [u32; 16] =
        core::array::from_fn(|i| u32::from_le_bytes(block[4 * i..][..4].try_into().unwrap()));
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(&h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u32;
    v[13] ^= (t >> 32) as u32;
    if last {
        v[14] ^= u32::MAX;
    }
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
    for s in SIGMA {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    core::array::from_fn(|i| h[i] ^ v[i] ^ v[i + 8])
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

/// One role `h_r(x; y z)`: chaining value `x`, block `y || z`, counter `r`, final flag clear.
fn h(role: u64, x: [u32; 8], y: [u32; 8], z: [u32; 8]) -> [u32; 8] {
    let mut block = [0u8; 64];
    block[..32].copy_from_slice(&bytes(y));
    block[32..].copy_from_slice(&bytes(z));
    reference_compress(x, &block, role, false)
}

/// Lane-wise XOR of two blocks.
fn xor(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
    core::array::from_fn(|i| a[i] ^ b[i])
}

/// T8 on a record of `7k + 1` blocks, exactly as the construction states it.
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
fn reference_compression_matches_blake2s() {
    // Invariant: the oracle is BLAKE2s's compression, pinned through two public digests.
    //
    //     40 bytes    one final block, counter 40
    //     100 bytes   one plain block with counter 64, then a final block with counter 100
    let mut h0 = [
        0x6a09_e667u32,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    h0[0] ^= 0x0101_0020;
    let digest =
        |m: &[u8]| -> [u8; 32] { <blake2::Blake2s256 as blake2::Digest>::digest(m).into() };

    let m = data(40, 1);
    let mut block = [0u8; 64];
    block[..40].copy_from_slice(&m);
    assert_eq!(bytes(reference_compress(h0, &block, 40, true)), digest(&m));

    let m = data(100, 2);
    let mut last = [0u8; 64];
    last[..36].copy_from_slice(&m[64..]);
    let state = reference_compress(
        reference_compress(h0, m[..64].try_into().unwrap(), 64, false),
        &last,
        100,
        true,
    );
    assert_eq!(bytes(state), digest(&m));
}

#[test]
fn record_lengths() {
    // Invariant: a record is 32 + 224 k bytes with k >= 1.
    assert_eq!(stages(256), Some(1));
    assert_eq!(stages(480), Some(2));
    for len in [0, 32, 224, 255, 257, 288, 512] {
        assert_eq!(stages(len), None, "len {len}");
    }
}

#[test]
fn known_answers() {
    // Invariant: the digests are fixed by the construction, its roles and BLAKE2s's word order.
    //
    // Fixture state: records whose byte i is i mod 256, of one and two stages.
    let one: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let two: Vec<u8> = (0..480).map(|i| i as u8).collect();
    assert_eq!(T8Blake2s.hash_slice(&one), reference_t8(&one));
    assert_eq!(T8Blake2s.hash_slice(&two), reference_t8(&two));
    assert_eq!(hex(&reference_t8(&one)), KAT_ONE_STAGE);
    assert_eq!(hex(&reference_t8(&two)), KAT_TWO_STAGES);
}

/// T8 of the bytes 0, 1, ..., 255.
const KAT_ONE_STAGE: &str = "2c0dd5a6e48e0bc3f4bf210f27d5923e9160c71876af602f7206bf8da5b8c0c6";

/// T8 of the bytes 0, 1, ..., 255, 0, ..., 223.
const KAT_TWO_STAGES: &str = "d4c93ffbfe92c8773431b168d33c7c044bc9ea9e6bd3f33be6775a3ad6c5cbfc";

#[test]
fn single_record_paths_match_the_reference() {
    // Invariant: the slice, iterator and piecewise paths all compute the same T8.
    for k in 1..=4 {
        let len = record_len(k);
        for record in [data(len, k as u64), vec![0u8; len], vec![0xFF; len]] {
            let expected = reference_t8(&record);
            assert_eq!(T8Blake2s.hash_slice(&record), expected, "k {k}");
            assert_eq!(
                T8Blake2s.hash_iter(record.iter().copied()),
                expected,
                "k {k}"
            );
            for piece in [1, 7, 31, 33, 225] {
                assert_eq!(
                    T8Blake2s.hash_iter_slices(record.chunks(piece)),
                    expected,
                    "k {k} piece {piece}"
                );
            }
        }
    }
}

#[test]
fn batches_match_the_reference() {
    // Invariant: the batched path agrees with the reference on every batch shape.
    //
    // Fixture state: counts around every group size: 8 (SSE2, NEON), 16 (AVX2), 32 (AVX-512).
    for k in 1..=3 {
        let len = record_len(k);
        for count in [0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100] {
            let input = data(len * count, (k * 1000 + count) as u64);
            let mut out = vec![[0u8; 32]; count];
            T8Blake2s.hash_many(&input, &mut out);
            for (i, (record, digest)) in input.chunks_exact(len).zip(&out).enumerate() {
                assert_eq!(
                    *digest,
                    reference_t8(record),
                    "k {k} count {count} record {i}"
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "32 + 224 k")]
fn batches_reject_other_lengths() {
    // Two records of 257 bytes each: not a T8 record length.
    let mut out = [[0u8; 32]; 2];
    T8Blake2s.hash_many(&[0u8; 514], &mut out);
}

proptest! {
    #[test]
    fn batches_match_one_record_at_a_time(count in 0usize..70, k in 1usize..=2, seed in any::<u64>()) {
        // Invariant: the batched digests are the single-record digests, in order.
        let len = record_len(k);
        let input = data(len * count, seed);
        let mut out = vec![[0u8; 32]; count];
        T8Blake2s.hash_many(&input, &mut out);
        for (record, digest) in input.chunks_exact(len).zip(&out) {
            prop_assert_eq!(*digest, T8Blake2s.hash_slice(record));
        }
    }
}
