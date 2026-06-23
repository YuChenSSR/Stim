//! Unit tests for the SIMD memory layer, mirroring cases from Stim's
//! `simd_bits.test.cc` / `simd_bit_table.test.cc`.

use rand::SeedableRng;
use rand_pcg::Pcg64;
use stim_core::mem::simd_bit_table::SimdBitTable;
use stim_core::mem::simd_bits::SimdBits;

#[test]
fn padding_is_at_least_256_bits() {
    let b = SimdBits::new(1);
    assert!(b.words().len() >= 4); // >= one 256-bit word
    let b = SimdBits::new(300);
    assert_eq!(b.words().len() % 4, 0);
    assert!(b.words().len() * 64 >= 300);
}

#[test]
fn get_set_xor_bits() {
    let mut b = SimdBits::new(200);
    assert!(!b.get(73));
    b.set(73, true);
    assert!(b.get(73));
    assert_eq!(b.popcnt(), 1);
    b.xor_bit(73, true);
    assert!(!b.get(73));
    assert_eq!(b.popcnt(), 0);
    b.xor_bit(199, true);
    assert!(b.get(199));
    assert_eq!(b.popcnt(), 1);
}

#[test]
fn bulk_xor_and_or_match_scalar() {
    let mut rng = Pcg64::seed_from_u64(7);
    for n in [1usize, 64, 200, 256, 257, 1000] {
        let mut a = SimdBits::new(n);
        let mut b = SimdBits::new(n);
        a.randomize(n, &mut rng);
        b.randomize(n, &mut rng);

        // Reference scalar results.
        let xor_ref: Vec<u64> = a
            .words()
            .iter()
            .zip(b.words())
            .map(|(x, y)| x ^ y)
            .collect();
        let and_ref: Vec<u64> = a
            .words()
            .iter()
            .zip(b.words())
            .map(|(x, y)| x & y)
            .collect();
        let or_ref: Vec<u64> = a
            .words()
            .iter()
            .zip(b.words())
            .map(|(x, y)| x | y)
            .collect();

        let mut t = a.clone();
        t.xor_assign(&b);
        assert_eq!(t.words(), &xor_ref[..], "xor n={n}");
        let mut t = a.clone();
        t.and_assign(&b);
        assert_eq!(t.words(), &and_ref[..], "and n={n}");
        let mut t = a.clone();
        t.or_assign(&b);
        assert_eq!(t.words(), &or_ref[..], "or n={n}");
    }
}

#[test]
fn xor_is_involutive() {
    let mut rng = Pcg64::seed_from_u64(11);
    let mut a = SimdBits::new(500);
    let b = {
        let mut x = SimdBits::new(500);
        x.randomize(500, &mut rng);
        x
    };
    a.randomize(500, &mut rng);
    let original = a.clone();
    a.xor_assign(&b);
    a.xor_assign(&b);
    assert_eq!(a, original);
}

#[test]
fn randomize_leaves_padding_zero() {
    let mut rng = Pcg64::seed_from_u64(3);
    let mut b = SimdBits::new(100);
    b.randomize(100, &mut rng);
    // Bits at/after 100 must be zero.
    for k in 100..b.words().len() * 64 {
        assert!(!b.get(k), "padding bit {k} should be zero");
    }
}

#[test]
fn for_each_set_bit_visits_exactly_the_set_bits() {
    let mut b = SimdBits::new(300);
    let expected = [0usize, 5, 63, 64, 200, 299];
    for &k in &expected {
        b.set(k, true);
    }
    let mut seen = Vec::new();
    b.for_each_set_bit(|k| seen.push(k));
    seen.sort_unstable();
    assert_eq!(seen, expected);
}

#[test]
fn swap_with_exchanges_contents() {
    let mut a = SimdBits::new(128);
    let mut b = SimdBits::new(128);
    a.set(3, true);
    b.set(70, true);
    a.swap_with(&mut b);
    assert!(b.get(3) && !b.get(70));
    assert!(a.get(70) && !a.get(3));
}

#[test]
fn table_two_rows_mut_distinct() {
    let mut t = SimdBitTable::new(4, 128);
    t.row_mut(0).set(1, true);
    t.row_mut(3).set(2, true);
    let (a, b) = t.two_rows_mut(0, 3);
    assert!(a.get(1));
    assert!(b.get(2));
    // Swap rows 0 and 3.
    a.swap_with(b);
    assert!(t.row(3).get(1));
    assert!(t.row(0).get(2));
}

#[test]
#[should_panic]
fn table_two_rows_mut_same_panics() {
    let mut t = SimdBitTable::new(4, 128);
    let _ = t.two_rows_mut(2, 2);
}
