//! Densely packed bit buffer with SIMD-friendly padding.
//!
//! Rust analogue of Stim's `simd_bits<W>` (`src/stim/mem/simd_bits.h`/`.inl`).
//! Like Stim, the backing store is padded up to a whole number of 256-bit
//! words, so the smallest buffer is 256 bits and word-at-a-time loops never run
//! off the end. Unlike Stim, the representation is a plain `Vec<u64>` rather
//! than a `union` aliased three ways, so there is no strict-aliasing hazard.

use crate::mem::bitword::{self, LANES_PER_WORD};
use rand::RngCore;

#[derive(Clone, Debug)]
pub struct SimdBits {
    /// Intended (unpadded) number of bits. Tracked separately, as in Stim.
    num_bits: usize,
    /// Backing lanes. Length is always a multiple of `LANES_PER_WORD`.
    words: Vec<u64>,
}

impl SimdBits {
    /// Zero-initialized buffer with at least `min_bits` bits (rounded up to a
    /// whole 256-bit word).
    pub fn new(min_bits: usize) -> Self {
        let num_u64 = Self::padded_u64_count(min_bits);
        SimdBits {
            num_bits: min_bits,
            words: vec![0u64; num_u64],
        }
    }

    fn padded_u64_count(min_bits: usize) -> usize {
        let bits_per_word = 64 * LANES_PER_WORD; // 256
        let words = (min_bits + bits_per_word - 1) / bits_per_word;
        // At least one full SIMD word, matching Stim's 256-bit minimum.
        words.max(1) * LANES_PER_WORD
    }

    #[inline]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    #[inline]
    pub fn get(&self, k: usize) -> bool {
        (self.words[k >> 6] >> (k & 63)) & 1 != 0
    }

    #[inline]
    pub fn set(&mut self, k: usize, value: bool) {
        let w = k >> 6;
        let mask = 1u64 << (k & 63);
        if value {
            self.words[w] |= mask;
        } else {
            self.words[w] &= !mask;
        }
    }

    #[inline]
    pub fn xor_bit(&mut self, k: usize, value: bool) {
        if value {
            self.words[k >> 6] ^= 1u64 << (k & 63);
        }
    }

    /// All bits to zero.
    pub fn clear(&mut self) {
        for w in self.words.iter_mut() {
            *w = 0;
        }
    }

    /// `self ^= other` (bulk SIMD).
    #[inline]
    pub fn xor_assign(&mut self, other: &SimdBits) {
        bitword::xor_into(&mut self.words, &other.words);
    }

    /// `self &= other` (bulk SIMD).
    #[inline]
    pub fn and_assign(&mut self, other: &SimdBits) {
        bitword::and_into(&mut self.words, &other.words);
    }

    /// `self |= other` (bulk SIMD).
    #[inline]
    pub fn or_assign(&mut self, other: &SimdBits) {
        bitword::or_into(&mut self.words, &other.words);
    }

    /// Copies all lanes from `other` into `self`.
    pub fn copy_from(&mut self, other: &SimdBits) {
        debug_assert_eq!(self.words.len(), other.words.len());
        self.words.copy_from_slice(&other.words);
        self.num_bits = other.num_bits;
    }

    /// Swaps contents with `other` (Stim's `swap_with`, used by H).
    pub fn swap_with(&mut self, other: &mut SimdBits) {
        std::mem::swap(&mut self.words, &mut other.words);
        std::mem::swap(&mut self.num_bits, &mut other.num_bits);
    }

    /// Number of set bits.
    pub fn popcnt(&self) -> usize {
        bitword::popcount(&self.words)
    }

    /// Randomizes the first `num_bits` bits using `rng`; padding stays zero.
    pub fn randomize(&mut self, num_bits: usize, rng: &mut impl RngCore) {
        let full = num_bits / 64;
        for w in 0..full {
            self.words[w] = rng.next_u64();
        }
        let rem = num_bits & 63;
        if rem != 0 {
            let mask = (1u64 << rem) - 1;
            self.words[full] = rng.next_u64() & mask;
            for w in (full + 1)..self.words.len() {
                self.words[w] = 0;
            }
        } else {
            for w in full..self.words.len() {
                self.words[w] = 0;
            }
        }
    }

    /// Invokes `callback` with the index of every set bit.
    pub fn for_each_set_bit(&self, mut callback: impl FnMut(usize)) {
        for (w, &mut_word) in self.words.iter().enumerate() {
            let mut v = mut_word;
            while v != 0 {
                let b = v.trailing_zeros() as usize;
                callback(w * 64 + b);
                v &= v - 1;
            }
        }
    }
}

impl PartialEq for SimdBits {
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
    }
}
impl Eq for SimdBits {}
