//! Deterministic, bit-exact correctness tests for the frame simulator.
//!
//! These disable the anticommutation frame-randomization so a single shot's
//! Pauli frame evolves deterministically, then check it against the known
//! conjugation rules — the same approach used by the deterministic cases in
//! Stim's `frame_simulator.test.cc`.

use rand::SeedableRng;
use rand_pcg::Pcg64;
use stim_core::circuit::{Circuit, Gate, Instruction};
use stim_core::frame_simulator::FrameSimulator;

/// Applies one gate to a 1-shot frame and returns the resulting (xs, zs).
fn propagate(num_qubits: usize, xs: &[bool], zs: &[bool], inst: Instruction) -> (Vec<bool>, Vec<bool>) {
    let rng = Pcg64::seed_from_u64(0);
    let mut sim = FrameSimulator::new(num_qubits, 1, rng);
    sim.guarantee_anticommutation_via_frame_randomization = false;
    sim.set_frame(0, xs, zs);
    sim.do_instruction(&inst);
    sim.get_frame(0)
}

fn inst(gate: Gate, targets: &[u32]) -> Instruction {
    Instruction {
        gate,
        args: vec![],
        targets: targets.to_vec(),
    }
}

#[test]
fn hadamard_swaps_x_and_z() {
    // X -> Z
    let (xs, zs) = propagate(1, &[true], &[false], inst(Gate::H, &[0]));
    assert_eq!((xs, zs), (vec![false], vec![true]));
    // Z -> X
    let (xs, zs) = propagate(1, &[false], &[true], inst(Gate::H, &[0]));
    assert_eq!((xs, zs), (vec![true], vec![false]));
}

#[test]
fn s_sends_x_to_y() {
    // S: X -> Y, i.e. (x=1,z=0) -> (x=1,z=1).
    let (xs, zs) = propagate(1, &[true], &[false], inst(Gate::S, &[0]));
    assert_eq!((xs, zs), (vec![true], vec![true]));
    // S leaves Z fixed.
    let (xs, zs) = propagate(1, &[false], &[true], inst(Gate::S, &[0]));
    assert_eq!((xs, zs), (vec![false], vec![true]));
}

#[test]
fn pauli_gates_are_frame_noops() {
    for g in [Gate::X, Gate::Y, Gate::Z, Gate::I] {
        let (xs, zs) = propagate(1, &[true], &[true], inst(g, &[0]));
        assert_eq!((xs, zs), (vec![true], vec![true]), "{g:?}");
    }
}

#[test]
fn cx_propagates_x_forward_and_z_backward() {
    // X on control spreads to target: (x0=1) -> x0=1, x1=1.
    let (xs, zs) = propagate(2, &[true, false], &[false, false], inst(Gate::Cx, &[0, 1]));
    assert_eq!(xs, vec![true, true]);
    assert_eq!(zs, vec![false, false]);

    // Z on target spreads back to control: (z1=1) -> z0=1, z1=1.
    let (xs, zs) = propagate(2, &[false, false], &[false, true], inst(Gate::Cx, &[0, 1]));
    assert_eq!(xs, vec![false, false]);
    assert_eq!(zs, vec![true, true]);
}

#[test]
fn cz_is_symmetric() {
    // X on qubit 0 induces Z on qubit 1 and vice versa.
    let (xs, zs) = propagate(2, &[true, false], &[false, false], inst(Gate::Cz, &[0, 1]));
    assert_eq!(xs, vec![true, false]);
    assert_eq!(zs, vec![false, true]);

    let (xs, zs) = propagate(2, &[false, true], &[false, false], inst(Gate::Cz, &[0, 1]));
    assert_eq!(xs, vec![false, true]);
    assert_eq!(zs, vec![true, false]);
}

#[test]
fn swap_exchanges_frames() {
    let (xs, zs) = propagate(2, &[true, false], &[false, true], inst(Gate::Swap, &[0, 1]));
    assert_eq!(xs, vec![false, true]);
    assert_eq!(zs, vec![true, false]);
}

#[test]
fn measurement_records_x_component() {
    // With randomization off, measuring in Z records the X frame component.
    let rng = Pcg64::seed_from_u64(0);
    let mut sim = FrameSimulator::new(2, 1, rng);
    sim.guarantee_anticommutation_via_frame_randomization = false;
    sim.set_frame(0, &[true, false], &[false, false]);
    sim.do_instruction(&inst(Gate::M, &[0, 1]));
    let recs = sim.measurement_flips();
    assert_eq!(recs.len(), 2);
    assert!(recs[0].get(0), "qubit 0 had an X frame -> measurement flipped");
    assert!(!recs[1].get(0), "qubit 1 had no X frame -> no flip");
}

#[test]
fn ghz_then_measure_is_correlated() {
    // H 0; CX 0 1; CX 0 2 builds a GHZ-style frame relationship. Injecting an X
    // on qubit 0 before the entangling gates flips all three measurements
    // together (deterministic with randomization off).
    let rng = Pcg64::seed_from_u64(0);
    let mut sim = FrameSimulator::new(3, 1, rng);
    sim.guarantee_anticommutation_via_frame_randomization = false;
    sim.set_frame(0, &[true, false, false], &[false, false, false]); // X on qubit 0
    sim.do_instruction(&inst(Gate::Cx, &[0, 1]));
    sim.do_instruction(&inst(Gate::Cx, &[0, 2]));
    sim.do_instruction(&inst(Gate::M, &[0, 1, 2]));
    let recs = sim.measurement_flips();
    assert!(recs[0].get(0) && recs[1].get(0) && recs[2].get(0));
}

#[test]
fn parser_roundtrip_and_run() {
    let text = "
        # a small repetition-code round
        R 0 1 2 3 4
        TICK
        CX 0 1 2 3
        CX 2 1 4 3
        MR 1 3
        QUBIT_COORDS(0, 0) 0
    ";
    let c = Circuit::from_text(text).expect("parse");
    assert_eq!(c.num_qubits(), 5);
    assert_eq!(c.num_measurements(), 2);

    // Runs without panicking and produces the right record count.
    let rng = Pcg64::seed_from_u64(1);
    let mut sim = FrameSimulator::new(c.num_qubits(), 8, rng);
    sim.do_circuit(&c);
    assert_eq!(sim.measurement_flips().len(), 2);
}

#[test]
fn parser_supports_repeat_blocks() {
    let text = "
        R 0
        REPEAT 5 {
            M 0
        }
    ";
    let c = Circuit::from_text(text).expect("parse");
    assert_eq!(c.num_measurements(), 5);
}

#[test]
fn parser_rejects_unsupported_gate() {
    assert!(Circuit::from_text("FOOBAR 0 1").is_err());
}
