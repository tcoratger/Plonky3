//! Two SHA-256 compressions at once, as two interleaved SHA-NI streams.
//!
//! `sha256rnds2` has a longer latency than its issue interval.
//!
//! One stream is a pure dependency chain, so it leaves the unit idle between its rounds.
//!
//! A second, independent stream fills those gaps, and costs little more than the first.
//!
//! The two independent calls of a T8 stage are such a pair.

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_loadu_si128, _mm_set_epi32, _mm_sha256msg1_epu32,
    _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_shuffle_epi8, _mm_shuffle_epi32,
    _mm_storeu_si128,
};

/// Eight state words `a .. h`, or eight big-endian block words.
type Words = [u32; 8];

/// The 64 round constants of FIPS 180-4 section 4.2.2.
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Byte permutation reversing each 32-bit lane: SHA-256 reads words big-endian.
const REVERSE_BYTES: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];

/// Whether the running CPU has SHA-NI and SSE4.1.
#[inline]
pub(crate) fn supported() -> bool {
    cpufeatures::new!(has_sha_ni, "sha", "sse4.1");
    has_sha_ni::get()
}

/// One stream: the state in SHA-NI lane order, and the four live schedule vectors.
struct Stream {
    /// Words `f e b a`, from the bottom lane up.
    abef: __m128i,
    /// Words `h g d c`, from the bottom lane up.
    cdgh: __m128i,
    /// `W[4g .. 4g + 4]` for the last four round groups.
    w: [__m128i; 4],
}

/// Two SHA-256 compressions of full 64-byte blocks, from any two chaining values.
///
/// Returns the two new chaining values, feed-forward included.
///
/// # Safety
///
/// The running CPU has SHA-NI and SSE4.1.
#[target_feature(enable = "sha,sse4.1")]
pub(crate) unsafe fn compress_pair(
    a: (&Words, &[u8; 64]),
    b: (&Words, &[u8; 64]),
) -> (Words, Words) {
    // SAFETY: `[u8; 16]` and `__m128i` have the same size, and every bit pattern is valid.
    let reverse = unsafe { core::mem::transmute::<[u8; 16], __m128i>(REVERSE_BYTES) };

    // State words into SHA-NI order, and the block as four big-endian vectors.
    let open = |(h, block): (&Words, &[u8; 64])| Stream {
        abef: _mm_set_epi32(h[0] as i32, h[1] as i32, h[4] as i32, h[5] as i32),
        cdgh: _mm_set_epi32(h[2] as i32, h[3] as i32, h[6] as i32, h[7] as i32),
        // SAFETY: four 16-byte reads inside the 64-byte block.
        w: core::array::from_fn(|k| unsafe {
            _mm_shuffle_epi8(_mm_loadu_si128(block.as_ptr().add(16 * k).cast()), reverse)
        }),
    };
    let mut s = [open(a), open(b)];
    let entry = [(s[0].abef, s[0].cdgh), (s[1].abef, s[1].cdgh)];

    // Sixteen groups of four rounds; both streams advance in the same group before the next.
    for g in 0..16 {
        // SAFETY: `K` has 64 words, and `g < 16`, so the 16-byte read is in bounds.
        let k = unsafe { _mm_loadu_si128(K.as_ptr().add(4 * g).cast()) };
        for s in &mut s {
            // From group 4 on, extend the schedule in place of its oldest vector.
            //
            //     W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]
            if g >= 4 {
                let [oldest, older, recent, newest] = [
                    s.w[g % 4],
                    s.w[(g + 1) % 4],
                    s.w[(g + 2) % 4],
                    s.w[(g + 3) % 4],
                ];
                let carried = _mm_alignr_epi8::<4>(newest, recent);
                s.w[g % 4] = _mm_sha256msg2_epu32(
                    _mm_add_epi32(_mm_sha256msg1_epu32(oldest, older), carried),
                    newest,
                );
            }

            // Two rounds from the low half of W + K, then two from the high half.
            let wk = _mm_add_epi32(s.w[g % 4], k);
            s.cdgh = _mm_sha256rnds2_epu32(s.cdgh, s.abef, wk);
            s.abef = _mm_sha256rnds2_epu32(s.abef, s.cdgh, _mm_shuffle_epi32::<0x0e>(wk));
        }
    }

    // Feed-forward, then back to state words `a .. h`.
    let close = |s: &Stream, (abef, cdgh): (__m128i, __m128i)| -> Words {
        let (mut x, mut y) = ([0u32; 4], [0u32; 4]);
        // SAFETY: each store writes 16 bytes inside a four-word array.
        unsafe {
            _mm_storeu_si128(x.as_mut_ptr().cast(), _mm_add_epi32(s.abef, abef));
            _mm_storeu_si128(y.as_mut_ptr().cast(), _mm_add_epi32(s.cdgh, cdgh));
        }
        // x = [f e b a] and y = [h g d c], from the bottom lane up.
        [x[3], x[2], y[3], y[2], x[1], x[0], y[1], y[0]]
    };
    (close(&s[0], entry[0]), close(&s[1], entry[1]))
}
