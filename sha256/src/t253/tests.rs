use alloc::vec;
use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;
use proptest::prelude::*;
use sha2::Digest;

use super::{BLOCK_BYTES, ROLES, STAGE_STRIDE, T253Sha256, stages};

/// The SHA-256 compression function of FIPS 180-4 section 6.2.2, written with no shared code.
fn reference_compress(state: [u32; 8], block: &[u8; 64]) -> [u32; 8] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut w = [0u32; 64];
    for t in 0..16 {
        w[t] = u32::from_be_bytes(block[4 * t..][..4].try_into().unwrap());
    }
    for t in 16..64 {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16]
            .wrapping_add(s0)
            .wrapping_add(w[t - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
    for t in 0..64 {
        let t1 = h
            .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
            .wrapping_add((e & f) ^ (!e & g))
            .wrapping_add(K[t])
            .wrapping_add(w[t]);
        let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
            .wrapping_add((a & b) ^ (a & c) ^ (b & c));
        (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
    }
    let out = [a, b, c, d, e, f, g, h];
    core::array::from_fn(|i| state[i].wrapping_add(out[i]))
}

/// Eight big-endian words of a 32-byte block.
fn words(bytes: &[u8]) -> [u32; 8] {
    core::array::from_fn(|i| u32::from_be_bytes(bytes[4 * i..][..4].try_into().unwrap()))
}

/// The 32 big-endian bytes of eight words.
fn bytes(words: [u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in words.iter().enumerate() {
        out[4 * i..][..4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

/// A tagged chaining value: the role byte, then 31 payload bytes.
fn tagged(role: u8, payload: &[u8]) -> [u32; 8] {
    let mut cv = [0u8; 32];
    cv[0] = role;
    cv[1..].copy_from_slice(payload);
    words(&cv)
}

/// Lane-wise XOR of two blocks.
fn xor(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
    core::array::from_fn(|i| a[i] ^ b[i])
}

/// T253 on a record of `32 + 221 k` bytes, exactly as the construction states it.
///
/// ```text
///     a  = h(0x01 || u_1;  s || y_3)
///     b  = h(0x02 || u_4;  m_2)
///     s' = h(0x03 || u_8;  (a ^ z_7) || (b ^ z_7)) ^ z_7
/// ```
fn reference_t253(record: &[u8]) -> [u8; 32] {
    assert_eq!((record.len() - 32) % 221, 0);
    let mut s = words(&record[..32]);
    for fresh in record[32..].as_chunks::<221>().0 {
        let mut m1 = [0u8; 64];
        m1[..32].copy_from_slice(&bytes(s));
        m1[32..].copy_from_slice(&fresh[31..63]);
        let a = reference_compress(tagged(ROLES[0], &fresh[..31]), &m1);
        let b = reference_compress(
            tagged(ROLES[1], &fresh[63..94]),
            fresh[94..158].try_into().unwrap(),
        );
        let z7 = words(&fresh[158..190]);
        let mut m3 = [0u8; 64];
        m3[..32].copy_from_slice(&bytes(xor(a, z7)));
        m3[32..].copy_from_slice(&bytes(xor(b, z7)));
        s = xor(
            reference_compress(tagged(ROLES[2], &fresh[190..221]), &m3),
            z7,
        );
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
fn reference_compression_matches_sha256() {
    // Invariant: the oracle is SHA-256's compression, pinned through two public digests.
    //
    //     "abc"      one padded block
    //     56 bytes   the length spills into a second, padding-only block
    const IV: [u32; 8] = crate::H256_256;
    let mut block = [0u8; 64];
    block[..3].copy_from_slice(b"abc");
    block[3] = 0x80;
    block[63] = 24;
    assert_eq!(
        bytes(reference_compress(IV, &block)),
        <[u8; 32]>::from(sha2::Sha256::digest(b"abc"))
    );

    let m = data(56, 1);
    let mut first = [0u8; 64];
    first[..56].copy_from_slice(&m);
    first[56] = 0x80;
    let mut second = [0u8; 64];
    second[62..].copy_from_slice(&(56u16 * 8).to_be_bytes());
    let state = reference_compress(reference_compress(IV, &first), &second);
    assert_eq!(bytes(state), <[u8; 32]>::from(sha2::Sha256::digest(&m)));
}

#[test]
fn record_lengths() {
    // Invariant: a record is 32 + 221 k bytes with k >= 1.
    assert_eq!(stages(253), Some(1));
    assert_eq!(stages(474), Some(2));
    for len in [0, 32, 221, 252, 254, 256, 480] {
        assert_eq!(stages(len), None, "len {len}");
    }
}

#[test]
fn known_answers() {
    // Invariant: the digests are fixed by the construction, its role tags and SHA-256's word order.
    //
    // Fixture state: records whose byte i is i mod 256, of one and two stages.
    let one: Vec<u8> = (0..253).map(|i| i as u8).collect();
    let two: Vec<u8> = (0..474).map(|i| i as u8).collect();
    assert_eq!(T253Sha256.hash_slice(&one), reference_t253(&one));
    assert_eq!(T253Sha256.hash_slice(&two), reference_t253(&two));
    assert_eq!(hex(&reference_t253(&one)), KAT_ONE_STAGE);
    assert_eq!(hex(&reference_t253(&two)), KAT_TWO_STAGES);
}

/// T253 of the bytes 0, 1, ..., 252.
const KAT_ONE_STAGE: &str = "47a8fab9cfbac107c0466af27f5271717bcf9903c813006291a9edc46bf73e04";

/// T253 of the bytes 0, 1, ..., 255, 0, ..., 217.
const KAT_TWO_STAGES: &str = "0cbab543da5ab58a36ee550cf936b80e5c0377dac6941eee92826c4b9a58f28c";

#[test]
fn single_record_paths_match_the_reference() {
    // Invariant: the slice, iterator and piecewise paths all compute the same T253.
    for k in 1..=4 {
        let len = record_len(k);
        for record in [data(len, k as u64), vec![0u8; len], vec![0xFF; len]] {
            let expected = reference_t253(&record);
            assert_eq!(T253Sha256.hash_slice(&record), expected, "k {k}");
            assert_eq!(
                T253Sha256.hash_iter(record.iter().copied()),
                expected,
                "k {k}"
            );
            for piece in [1, 7, 31, 33, 225] {
                assert_eq!(
                    T253Sha256.hash_iter_slices(record.chunks(piece)),
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
    // Fixture state: counts on both sides of the one-at-a-time cutoff, the four-stream group and the 32-lane group.
    for k in 1..=3 {
        let len = record_len(k);
        for count in [
            0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100,
        ] {
            let input = data(len * count, (k * 1000 + count) as u64);
            let mut out = vec![[0u8; 32]; count];
            T253Sha256.hash_many(&input, &mut out);
            for (i, (record, digest)) in input.chunks_exact(len).zip(&out).enumerate() {
                assert_eq!(
                    *digest,
                    reference_t253(record),
                    "k {k} count {count} record {i}"
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "32 + 221 k")]
fn batches_reject_other_lengths() {
    // Two records of 256 bytes each: a T8 length, not a T253 one.
    let mut out = [[0u8; 32]; 2];
    T253Sha256.hash_many(&[0u8; 512], &mut out);
}

proptest! {
    #[test]
    fn batches_match_one_record_at_a_time(count in 0usize..70, k in 1usize..=2, seed in any::<u64>()) {
        // Invariant: the batched digests are the single-record digests, in order.
        let len = record_len(k);
        let input = data(len * count, seed);
        let mut out = vec![[0u8; 32]; count];
        T253Sha256.hash_many(&input, &mut out);
        for (record, digest) in input.chunks_exact(len).zip(&out) {
            prop_assert_eq!(*digest, T253Sha256.hash_slice(record));
        }
    }
}

#[test]
fn roles_never_meet_the_node_hash() {
    // Invariant: every call's chaining value starts with a role tag.
    //
    // The tree's node hash starts from the SHA-256 initial value instead, whose top byte is 0x6a.
    //
    //     roles 0x01 0x02 0x03   node 0x6a   -> no input of one is an input of another
    let node_top = (crate::H256_256[0] >> 24) as u8;
    assert!(!ROLES.contains(&node_top));
    assert!(ROLES[0] != ROLES[1] && ROLES[1] != ROLES[2] && ROLES[0] != ROLES[2]);
}
