//! Statistical correctness of the noise channels. The frame simulator's RNG
//! differs from Stim's, so these validate sampled *rates* against the analytic
//! expectation rather than bit-for-bit against C++.

use stim_core::{measurement_flip_rates, Circuit};

const SHOTS: usize = 400_000;
const TOL: f64 = 0.01;

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < TOL,
        "{what}: got {actual:.4}, expected {expected:.4}"
    );
}

#[test]
fn x_error_flip_rate_matches_probability() {
    let c = Circuit::from_text("R 0\nX_ERROR(0.1) 0\nM 0").unwrap();
    let rates = measurement_flip_rates(&c, SHOTS, 12345);
    assert_close(rates[0], 0.1, "X_ERROR(0.1) Z-measurement flip rate");
}

#[test]
fn z_error_does_not_flip_z_measurement() {
    let c = Circuit::from_text("R 0\nZ_ERROR(0.5) 0\nM 0").unwrap();
    let rates = measurement_flip_rates(&c, SHOTS, 999);
    assert_close(rates[0], 0.0, "Z_ERROR has no effect on Z measurement");
}

#[test]
fn y_error_flips_z_measurement() {
    let c = Circuit::from_text("R 0\nY_ERROR(0.2) 0\nM 0").unwrap();
    let rates = measurement_flip_rates(&c, SHOTS, 77);
    assert_close(rates[0], 0.2, "Y_ERROR flips Z measurement");
}

#[test]
fn depolarize1_flips_z_measurement_two_thirds_of_the_time() {
    // DEPOLARIZE1(p) applies a uniform non-identity Pauli with prob p. Two of
    // the three (X, Y) flip a Z measurement, so the rate is p * 2/3.
    let p = 0.3;
    let c = Circuit::from_text(&format!("R 0\nDEPOLARIZE1({p}) 0\nM 0")).unwrap();
    let rates = measurement_flip_rates(&c, SHOTS, 4242);
    assert_close(rates[0], p * 2.0 / 3.0, "DEPOLARIZE1 Z-flip rate");
}

#[test]
fn reset_then_hadamard_measures_randomly() {
    // R 0; H 0 prepares |+>, and a Z-basis measurement of |+> is uniformly
    // random. This exercises the anticommutation frame-randomization: the reset
    // seeds a random Z component which H rotates into the measured X component.
    let c = Circuit::from_text("R 0\nH 0\nM 0").unwrap();
    let rates = measurement_flip_rates(&c, SHOTS, 8);
    assert_close(rates[0], 0.5, "measuring |+> in Z basis is random");
}

#[test]
fn cx_propagates_x_error_to_two_measurements() {
    // An X error on the control of a CX flips both qubits' Z measurements
    // identically. Both rates should equal the error probability.
    let c = Circuit::from_text("R 0 1\nX_ERROR(0.15) 0\nCX 0 1\nM 0 1").unwrap();
    let rates = measurement_flip_rates(&c, SHOTS, 2024);
    assert_close(rates[0], 0.15, "control measurement flip rate");
    assert_close(rates[1], 0.15, "target measurement flip rate (propagated)");
}

#[test]
fn repetition_code_detector_fires_at_data_error_rate() {
    // Distance-2 repetition-code parity check. A data X error between two
    // resets of the ancilla shows up as a flipped ancilla measurement.
    let c = Circuit::from_text(
        "
        R 0 1 2
        CX 0 1
        CX 2 1
        MR 1
        X_ERROR(0.05) 0
        CX 0 1
        CX 2 1
        MR 1
        ",
    )
    .unwrap();
    let rates = measurement_flip_rates(&c, SHOTS, 55);
    // First parity measurement: no error yet -> ~0.
    assert_close(rates[0], 0.0, "first parity measurement");
    // Second parity measurement detects the data error -> ~0.05.
    assert_close(rates[1], 0.05, "second parity measurement detects error");
}
