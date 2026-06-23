//! Low-level bulk bitwise operations over slices of 64-bit words.
//!
//! This is the Rust analogue of Stim's `bitword<W>` SIMD abstraction
//! (`src/stim/mem/bitword.h`, `bitword_256_avx.h`, `bitword_64.h`). Stim
//! selects a `__m256i`/`__m128i`/`uint64_t` backed word type at compile time
//! and runs every hot loop a word at a time. Here we keep the data as a flat
//! slice of `u64` lanes (padded to a multiple of 4, i.e. 256 bits, exactly like
//! Stim's minimum allocation) and process it either with hand-written AVX2
//! intrinsics or a portable scalar loop.
//!
//! The C++ version aliases one buffer as `u8*` / `u64*` / `__m256i*` through a
//! `union` and relies on `-fno-strict-aliasing`. We avoid that unsoundness:
//! the canonical representation is always `[u64]`, and the AVX2 path uses
//! unaligned loads/stores so no alignment invariant has to be upheld by callers.

/// Number of u64 lanes per 256-bit SIMD word. All buffers are padded to a
/// multiple of this so the bulk loops never read or write out of bounds.
pub const LANES_PER_WORD: usize = 4;

/// `dst[i] ^= src[i]` for every lane.
#[inline]
pub fn xor_into(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(all(feature = "simd256", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 is available; the helper only touches `dst`/`src`.
            unsafe { return avx2::xor_into(dst, src) };
        }
    }
    scalar::xor_into(dst, src);
}

/// `dst[i] &= src[i]` for every lane.
#[inline]
pub fn and_into(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(all(feature = "simd256", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return avx2::and_into(dst, src) };
        }
    }
    scalar::and_into(dst, src);
}

/// `dst[i] |= src[i]` for every lane.
#[inline]
pub fn or_into(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(all(feature = "simd256", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return avx2::or_into(dst, src) };
        }
    }
    scalar::or_into(dst, src);
}

/// Population count over all lanes.
#[inline]
pub fn popcount(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

mod scalar {
    #[inline]
    pub fn xor_into(dst: &mut [u64], src: &[u64]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d ^= *s;
        }
    }
    #[inline]
    pub fn and_into(dst: &mut [u64], src: &[u64]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d &= *s;
        }
    }
    #[inline]
    pub fn or_into(dst: &mut [u64], src: &[u64]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d |= *s;
        }
    }
}

#[cfg(all(feature = "simd256", target_arch = "x86_64"))]
mod avx2 {
    use super::LANES_PER_WORD;
    use core::arch::x86_64::*;

    /// Applies a per-256-bit-word AVX2 op to `dst`, mirroring Stim's word loop.
    /// Handles any trailing lanes (when len isn't a multiple of 4) with scalar
    /// code so the helper is correct for unpadded slices too.
    #[target_feature(enable = "avx2")]
    unsafe fn each_word(
        dst: &mut [u64],
        src: &[u64],
        op: unsafe fn(__m256i, __m256i) -> __m256i,
        scalar_op: fn(u64, u64) -> u64,
    ) {
        let n = dst.len();
        let chunks = n / LANES_PER_WORD;
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        for c in 0..chunks {
            let off = (c * LANES_PER_WORD) as isize;
            let a = _mm256_loadu_si256(dp.offset(off) as *const __m256i);
            let b = _mm256_loadu_si256(sp.offset(off) as *const __m256i);
            _mm256_storeu_si256(dp.offset(off) as *mut __m256i, op(a, b));
        }
        for i in (chunks * LANES_PER_WORD)..n {
            *dst.get_unchecked_mut(i) = scalar_op(*dst.get_unchecked(i), *src.get_unchecked(i));
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn xor_into(dst: &mut [u64], src: &[u64]) {
        each_word(dst, src, _mm256_xor_si256, |a, b| a ^ b);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn and_into(dst: &mut [u64], src: &[u64]) {
        each_word(dst, src, _mm256_and_si256, |a, b| a & b);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn or_into(dst: &mut [u64], src: &[u64]) {
        each_word(dst, src, _mm256_or_si256, |a, b| a | b);
    }
}
