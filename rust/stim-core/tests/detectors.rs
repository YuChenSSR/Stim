//! Tests for detector / observable sampling — the analogue of Stim's
//! `compile_detector_sampler`.

use stim_core::{sample_detectors, Circuit};

/// Two-round distance-3 repetition-code memory. Data qubits 0/2/4, ancillas
/// 1/3. `mid` is inserted between the two measurement rounds (e.g. an error on a
/// data qubit). Detector 0 compares ancilla 1 across rounds, detector 1 ancilla
/// 3. Observable 0 is the final data measurement of qubit 0.
fn rep_code(mid: &str) -> String {
    format!(
        "
        R 0 1 2 3 4
        CX 0 1 2 3
        CX 2 1 4 3
        MR 1 3
        {mid}
        CX 0 1 2 3
        CX 2 1 4 3
        MR 1 3
        DETECTOR rec[-2] rec[-4]
        DETECTOR rec[-1] rec[-3]
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        "
    )
}

#[test]
fn parser_counts_detectors_and_observables() {
    let c = Circuit::from_text(&rep_code("")).unwrap();
    assert_eq!(c.num_detectors(), 2);
    assert_eq!(c.num_observables(), 1);
    // rec[] targets must not inflate the qubit count.
    assert_eq!(c.num_qubits(), 5);
}

#[test]
fn detectors_are_quiet_without_noise() {
    let c = Circuit::from_text(&rep_code("")).unwrap();
    let s = sample_detectors(&c, 2000, 1);
    assert!(
        s.detectors.iter().all(|row| row.iter().all(|&d| !d)),
        "noiseless detectors must never fire"
    );
}

#[test]
fn deterministic_data_error_lights_one_detector() {
    // A definite X on data qubit 0 between rounds flips ancilla 1's parity, so
    // detector 0 fires every shot; ancilla 3 (parity of data 2,4) is untouched.
    let c = Circuit::from_text(&rep_code("X 0")).unwrap();
    let s = sample_detectors(&c, 2000, 1);
    assert!(s.detectors.iter().all(|row| row[0]), "detector 0 must fire");
    assert!(s.detectors.iter().all(|row| !row[1]), "detector 1 must stay quiet");
}

#[test]
fn x_error_lights_detector_at_its_rate() {
    let c = Circuit::from_text(&rep_code("X_ERROR(0.05) 0")).unwrap();
    let shots = 200_000;
    let s = sample_detectors(&c, shots, 7);
    let fired = s.detectors.iter().filter(|row| row[0]).count();
    let rate = fired as f64 / shots as f64;
    assert!((rate - 0.05).abs() < 0.01, "detector rate {rate:.4}");
    // The data error also flips the final observable measurement at the same rate.
    let obs = s.observables.iter().filter(|row| row[0]).count();
    let obs_rate = obs as f64 / shots as f64;
    assert!((obs_rate - 0.05).abs() < 0.01, "observable rate {obs_rate:.4}");
}

#[test]
fn detector_picks_up_measurement_record_lookback_order() {
    // rec[-1] is the most recent measurement. A single qubit measured twice with
    // an X flip in between yields a detector that always fires.
    let c = Circuit::from_text(
        "
        R 0
        M 0
        X 0
        M 0
        DETECTOR rec[-1] rec[-2]
        ",
    )
    .unwrap();
    let s = sample_detectors(&c, 500, 3);
    assert!(s.detectors.iter().all(|row| row[0]));
}
