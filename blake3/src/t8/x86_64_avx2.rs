//! Two BLAKE3 compressions at once, one per 128-bit half of each AVX2 register.
//!
//! The one-message compression keeps its state as four rows of four words, one row per register:
//!
//! ```text
//!     row 0   h_0 h_1 h_2 h_3        row 2   IV_0 IV_1 IV_2 IV_3
//!     row 1   h_4 h_5 h_6 h_7        row 3   t_lo t_hi len  flags
//! ```
//!
//! A 256-bit register holds that row for two messages, one in each half:
//!
//! ```text
//!     [ row of message A | row of message B ]
//! ```
//!
//! Every shuffle, blend and unpack below works within each half, so the two messages never mix.
//!
//! Two independent chains then share one instruction stream, and cost about one.

use core::arch::x86_64::*;

/// Chaining value or 32-byte block, as eight words.
type Words = [u32; 8];

/// The BLAKE3 initialization vector, the second half's first row.
const IV: [u32; 4] = [0x6A09_E667, 0xBB67_AE85, 0x3C6E_F372, 0xA54F_F53A];

/// One compression's inputs: chaining value, 64-byte block, counter and flags, with a full block.
pub(crate) struct Call<'a> {
    /// The chaining value.
    pub(crate) cv: Words,
    /// The 64-byte block.
    pub(crate) block: &'a [u8; 64],
    /// The counter word.
    pub(crate) counter: u64,
    /// The flag bits.
    pub(crate) flags: u32,
}

/// Whether the running CPU has AVX2.
#[inline]
pub(crate) fn supported() -> bool {
    cpufeatures::new!(has_avx2, "avx2");
    has_avx2::get()
}

/// Build `_mm_shuffle`-style immediates from four two-bit lane indices, highest first.
macro_rules! shuf {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

/// Two-source 32-bit shuffle within each half, through the float domain.
macro_rules! shuffle2 {
    ($a:expr, $b:expr, $imm:expr) => {
        _mm256_castps_si256(_mm256_shuffle_ps::<{ $imm }>(
            _mm256_castsi256_ps($a),
            _mm256_castsi256_ps($b),
        ))
    };
}

/// Rotate every 32-bit word right by `R`.
#[inline(always)]
fn rotr<const R: i32, const L: i32>(a: __m256i) -> __m256i {
    // SAFETY: the callers carry AVX2.
    unsafe { _mm256_or_si256(_mm256_srli_epi32::<R>(a), _mm256_slli_epi32::<L>(a)) }
}

/// Half of the G function on all four columns or diagonals: add, rotate by 16 and 12.
#[inline(always)]
fn g1(r: &mut [__m256i; 4], m: __m256i) {
    // SAFETY: the callers carry AVX2.
    unsafe {
        r[0] = _mm256_add_epi32(_mm256_add_epi32(r[0], m), r[1]);
        r[3] = rotr::<16, 16>(_mm256_xor_si256(r[3], r[0]));
        r[2] = _mm256_add_epi32(r[2], r[3]);
        r[1] = rotr::<12, 20>(_mm256_xor_si256(r[1], r[2]));
    }
}

/// The other half of the G function: add, rotate by 8 and 7.
#[inline(always)]
fn g2(r: &mut [__m256i; 4], m: __m256i) {
    // SAFETY: the callers carry AVX2.
    unsafe {
        r[0] = _mm256_add_epi32(_mm256_add_epi32(r[0], m), r[1]);
        r[3] = rotr::<8, 24>(_mm256_xor_si256(r[3], r[0]));
        r[2] = _mm256_add_epi32(r[2], r[3]);
        r[1] = rotr::<7, 25>(_mm256_xor_si256(r[1], r[2]));
    }
}

/// Rotate rows 0, 2 and 3 so the next G step works on diagonals.
///
/// Row 1 stays put, and the message loads compensate.
#[inline(always)]
fn diagonalize(r: &mut [__m256i; 4]) {
    // SAFETY: the callers carry AVX2.
    unsafe {
        r[0] = _mm256_shuffle_epi32::<{ shuf!(2, 1, 0, 3) }>(r[0]);
        r[3] = _mm256_shuffle_epi32::<{ shuf!(1, 0, 3, 2) }>(r[3]);
        r[2] = _mm256_shuffle_epi32::<{ shuf!(0, 3, 2, 1) }>(r[2]);
    }
}

/// Undo the diagonal rotation.
#[inline(always)]
fn undiagonalize(r: &mut [__m256i; 4]) {
    // SAFETY: the callers carry AVX2.
    unsafe {
        r[0] = _mm256_shuffle_epi32::<{ shuf!(0, 3, 2, 1) }>(r[0]);
        r[3] = _mm256_shuffle_epi32::<{ shuf!(1, 0, 3, 2) }>(r[3]);
        r[2] = _mm256_shuffle_epi32::<{ shuf!(2, 1, 0, 3) }>(r[2]);
    }
}

/// One later round: the fixed message permutation, applied with shuffles, then the round itself.
#[inline(always)]
fn permuted_round(r: &mut [__m256i; 4], m: &mut [__m256i; 4]) {
    // SAFETY: the callers carry AVX2.
    unsafe {
        let [m0, m1, m2, m3] = *m;
        let t0 =
            _mm256_shuffle_epi32::<{ shuf!(0, 3, 2, 1) }>(shuffle2!(m0, m1, shuf!(3, 1, 1, 2)));
        g1(r, t0);
        let t1 = _mm256_blend_epi16::<0xCC>(
            _mm256_shuffle_epi32::<{ shuf!(0, 0, 3, 3) }>(m0),
            shuffle2!(m2, m3, shuf!(3, 3, 2, 2)),
        );
        g2(r, t1);
        diagonalize(r);
        let t2 = _mm256_shuffle_epi32::<{ shuf!(1, 3, 2, 0) }>(_mm256_blend_epi16::<0xC0>(
            _mm256_unpacklo_epi64(m3, m1),
            m2,
        ));
        g1(r, t2);
        let t3 = _mm256_shuffle_epi32::<{ shuf!(0, 1, 3, 2) }>(_mm256_unpacklo_epi32(
            m2,
            _mm256_unpackhi_epi32(m1, m3),
        ));
        g2(r, t3);
        undiagonalize(r);
        *m = [t0, t1, t2, t3];
    }
}

/// Two full-block BLAKE3 compressions, `a` in the low halves and `b` in the high halves.
///
/// Returns the two new chaining values, the first eight output words of each.
///
/// # Safety
///
/// The running CPU has AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn compress_pair(a: &Call<'_>, b: &Call<'_>) -> (Words, Words) {
    // Two 128-bit halves side by side: message A low, message B high.
    let pair = |lo: __m128i, hi: __m128i| _mm256_set_m128i(hi, lo);
    // SAFETY: every load reads 16 bytes inside an eight-word array or a 64-byte block.
    let load = |p: *const u8| unsafe { _mm_loadu_si128(p.cast()) };
    let params = |c: &Call<'_>| {
        _mm_setr_epi32(
            c.counter as i32,
            (c.counter >> 32) as i32,
            64,
            c.flags as i32,
        )
    };
    let iv = _mm_setr_epi32(IV[0] as i32, IV[1] as i32, IV[2] as i32, IV[3] as i32);

    let mut r = [
        pair(load(a.cv.as_ptr().cast()), load(b.cv.as_ptr().cast())),
        // SAFETY: words 4 to 7 of an eight-word array.
        pair(
            load(unsafe { a.cv.as_ptr().add(4) }.cast()),
            load(unsafe { b.cv.as_ptr().add(4) }.cast()),
        ),
        pair(iv, iv),
        pair(params(a), params(b)),
    ];
    // SAFETY: offsets 0, 16, 32 and 48 of a 64-byte block.
    let msg = |k: usize| {
        pair(
            load(unsafe { a.block.as_ptr().add(16 * k) }),
            load(unsafe { b.block.as_ptr().add(16 * k) }),
        )
    };
    let [m0, m1, m2, m3] = [msg(0), msg(1), msg(2), msg(3)];

    // Round 1 gathers the message words from their natural order into the parallel groups.
    let t0 = shuffle2!(m0, m1, shuf!(2, 0, 2, 0));
    g1(&mut r, t0);
    let t1 = shuffle2!(m0, m1, shuf!(3, 1, 3, 1));
    g2(&mut r, t1);
    diagonalize(&mut r);
    let t2 = _mm256_shuffle_epi32::<{ shuf!(2, 1, 0, 3) }>(shuffle2!(m2, m3, shuf!(2, 0, 2, 0)));
    g1(&mut r, t2);
    let t3 = _mm256_shuffle_epi32::<{ shuf!(2, 1, 0, 3) }>(shuffle2!(m2, m3, shuf!(3, 1, 3, 1)));
    g2(&mut r, t3);
    undiagonalize(&mut r);

    // Rounds 2 to 7 permute the previous round's words the same way.
    let mut m = [t0, t1, t2, t3];
    for _ in 0..6 {
        permuted_round(&mut r, &mut m);
    }

    // The new chaining value is the XOR of the two halves of the working state.
    let lo = _mm256_xor_si256(r[0], r[2]);
    let hi = _mm256_xor_si256(r[1], r[3]);
    let mut out = ([0u32; 8], [0u32; 8]);
    // SAFETY: each store writes 16 bytes inside an eight-word array.
    unsafe {
        _mm_storeu_si128(out.0.as_mut_ptr().cast(), _mm256_castsi256_si128(lo));
        _mm_storeu_si128(out.0.as_mut_ptr().add(4).cast(), _mm256_castsi256_si128(hi));
        _mm_storeu_si128(out.1.as_mut_ptr().cast(), _mm256_extracti128_si256::<1>(lo));
        _mm_storeu_si128(
            out.1.as_mut_ptr().add(4).cast(),
            _mm256_extracti128_si256::<1>(hi),
        );
    }
    out
}
