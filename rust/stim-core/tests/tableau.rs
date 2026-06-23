//! Correctness tests for the CHP tableau simulator and the absolute sampler
//! built on top of it (tableau reference XOR frame flips).

use rand::SeedableRng;
use rand_pcg::Pcg64;
use stim_core::{sample, Circuit, TableauSimulator};

fn rng(seed: u64) -> Pcg64 {
    Pcg64::seed_from_u64(seed)
}

#[test]
fn deterministic_single_qubit_prep_and_measure() {
    // |0> measures 0; X|0> = |1> measures 1.
    let mut s = TableauSimulator::new(1, rng(1));
    assert!(!s.measure(0));

    let mut s = TableauSimulator::new(1, rng(1));
    s.x_gate(0);
    assert!(s.measure(0));

    // H then S^2 (=Z) then H = X, so |0> -> |1>.
    let mut s = TableauSimulator::new(1, rng(1));
    s.h(0);
    s.s(0);
    s.s(0);
    s.h(0);
    assert!(s.measure(0));
}

#[test]
fn bell_state_measurements_are_correlated() {
    // H 0; CX 0 1 then measuring both always gives equal results.
    for seed in 0..50 {
        let mut s = TableauSimulator::new(2, rng(seed));
        s.h(0);
        s.cnot(0, 1);
        let a = s.measure(0);
        let b = s.measure(1);
        assert_eq!(a, b, "Bell pair measurements must match (seed {seed})");
    }
}

#[test]
fn ghz_reference_sample_has_equal_bits() {
    let c = Circuit::from_text("H 0\nCX 0 1\nCX 0 2\nM 0 1 2").unwrap();
    for seed in 0..50 {
        let mut s = TableauSimulator::new(3, rng(seed));
        let m = s.sample_reference(&c);
        assert_eq!(m.len(), 3);
        assert!(m[0] == m[1] && m[1] == m[2], "GHZ bits must agree");
    }
}

#[test]
fn cz_matches_decomposition_via_phase_kickback() {
    // |+>|+> under CZ becomes a state where measuring qubit 0 in X basis after
    // CZ correlates with qubit 1. Simpler invariant: CZ is symmetric, so
    // CZ(0,1) and CZ(1,0) produce identical measurement statistics. Check that
    // H;H;CZ;H;H;M on |00> stays deterministic and equal both ways.
    let mut a = TableauSimulator::new(2, rng(0));
    a.h(0);
    a.h(1);
    a.cz(0, 1);
    a.h(0);
    a.h(1);

    let mut b = TableauSimulator::new(2, rng(0));
    b.h(0);
    b.h(1);
    // Manual CZ(1,0) via the same identity but swapped control/target.
    b.h(0);
    b.cnot(1, 0);
    b.h(0);
    b.h(0);
    b.h(1);

    assert_eq!(a.measure(0), b.measure(0));
    assert_eq!(a.measure(1), b.measure(1));
}

#[test]
fn swap_exchanges_computational_basis_states() {
    // Prepare |10>, SWAP, expect |01>.
    let mut s = TableauSimulator::new(2, rng(0));
    s.x_gate(0);
    s.swap(0, 1);
    assert!(!s.measure(0));
    assert!(s.measure(1));
}

#[test]
fn reset_forces_zero() {
    let mut s = TableauSimulator::new(1, rng(0));
    s.x_gate(0); // |1>
    s.reset(0);
    assert!(!s.measure(0), "after reset the qubit must be |0>");
}

#[test]
fn absolute_sampler_random_measurement_is_uniform() {
    // Measuring |+> in Z is uniformly random; the absolute sampler must produce
    // ~50% ones even though the reference picks one definite value.
    let c = Circuit::from_text("H 0\nM 0").unwrap();
    let shots = 200_000;
    let samples = sample(&c, shots, 42);
    let ones: usize = samples.iter().filter(|s| s[0]).count();
    let rate = ones as f64 / shots as f64;
    assert!((rate - 0.5).abs() < 0.01, "got {rate:.4}");
}

#[test]
fn absolute_sampler_ghz_keeps_correlations() {
    // Every shot must have all three measurement bits equal.
    let c = Circuit::from_text("H 0\nCX 0 1\nCX 0 2\nM 0 1 2").unwrap();
    let samples = sample(&c, 5_000, 7);
    for (i, s) in samples.iter().enumerate() {
        assert!(s[0] == s[1] && s[1] == s[2], "shot {i} broke GHZ correlation");
    }
    // And both 000 and 111 should actually occur.
    let any_ones = samples.iter().any(|s| s[0]);
    let any_zeros = samples.iter().any(|s| !s[0]);
    assert!(any_ones && any_zeros, "expected both outcomes to appear");
}

#[test]
fn absolute_sampler_deterministic_circuit_is_constant() {
    // X 0; M 0 always reads 1; reset-measure always 0.
    let c = Circuit::from_text("X 0\nM 0").unwrap();
    let samples = sample(&c, 1000, 3);
    assert!(samples.iter().all(|s| s[0]));

    let c = Circuit::from_text("R 0\nX 0\nX 0\nM 0").unwrap();
    let samples = sample(&c, 1000, 3);
    assert!(samples.iter().all(|s| !s[0]));
}
