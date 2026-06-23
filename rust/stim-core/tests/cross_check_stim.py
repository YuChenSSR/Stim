#!/usr/bin/env python3
"""Cross-validate stim-core (Rust) against the reference C++ Stim package.

For circuits whose noiseless measurements are deterministically 0 (reset-then-
measure repetition codes here), Stim's absolute measurement samples equal
stim-core's recorded measurement *flips*. Because the two use different RNGs we
compare per-measurement flip *rates*, not bit-for-bit samples.

Run from the repo root after `pip install stim`:
    python3 rust/stim-core/tests/cross_check_stim.py
"""
import subprocess
import sys
import tempfile
import pathlib

try:
    import stim
except ImportError:
    print("SKIP: python `stim` not installed (pip install stim)")
    sys.exit(0)

import numpy as np

SHOTS = 200_000
TOL = 0.01

# A small noisy repetition-code-style circuit. All measured qubits are reset
# before measuring, so the noiseless reference sample is all zeros.
CIRCUIT = """
R 0 1 2 3 4
X_ERROR(0.08) 0 2 4
CX 0 1 2 3
CX 2 1 4 3
MR 1 3
X_ERROR(0.03) 0 2 4
CX 0 1 2 3
CX 2 1 4 3
MR 1 3
"""

# A circuit with genuinely random measurements (no reset before the Hadamard
# basis change) AND noise. Stage 1 could not handle this; the stage-2 absolute
# sampler (tableau reference XOR frame flips) can.
ABSOLUTE_CIRCUIT = """
H 0
CX 0 1
CX 0 2
X_ERROR(0.1) 1
M 0 1 2
"""

ROOT = pathlib.Path(__file__).resolve().parents[3]
CRATE = ROOT / "rust"


def stim_rates(circuit_text):
    circuit = stim.Circuit(circuit_text)
    shots = circuit.compile_sampler().sample(SHOTS)
    return shots.mean(axis=0)


def write_example():
    example = CRATE / "stim-core" / "examples" / "rates.rs"
    example.parent.mkdir(parents=True, exist_ok=True)
    example.write_text(
        '''
use stim_core::{measurement_flip_rates, sample, Circuit};
fn main() {
    let mode = std::env::args().nth(1).unwrap();
    let text = std::env::args().nth(2).unwrap();
    let shots: usize = std::env::args().nth(3).unwrap().parse().unwrap();
    let c = Circuit::from_text(&text).unwrap();
    let rates: Vec<f64> = if mode == "flips" {
        measurement_flip_rates(&c, shots, 1234)
    } else {
        // Absolute sampler: per-measurement mean over all shots.
        let s = sample(&c, shots, 1234);
        let m = s[0].len();
        (0..m)
            .map(|j| s.iter().filter(|row| row[j]).count() as f64 / shots as f64)
            .collect()
    };
    let parts: Vec<String> = rates.iter().map(|r| format!("{r}")).collect();
    println!("{}", parts.join(","));
}
'''
    )


def rust_rates(mode, circuit_text):
    out = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "rates", "--",
         mode, circuit_text, str(SHOTS)],
        cwd=CRATE,
        capture_output=True,
        text=True,
        check=True,
    )
    line = out.stdout.strip().splitlines()[-1]
    return np.array([float(x) for x in line.split(",")])


def compare(name, circuit_text, mode):
    s = stim_rates(circuit_text)
    r = rust_rates(mode, circuit_text)
    print(f"\n[{name}]")
    print(f"  stim rates: {np.array2string(s, precision=4)}")
    print(f"  rust rates: {np.array2string(r, precision=4)}")
    if s.shape != r.shape:
        print(f"  FAIL: shape mismatch {s.shape} vs {r.shape}")
        return False
    diff = np.abs(s - r).max()
    ok = diff < TOL
    print(f"  max abs diff: {diff:.4f} (tol {TOL}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    write_example()
    ok = True
    # Stage 1: reference-zero circuit, compare flip rates to absolute samples.
    ok &= compare("flip-rate (reset-then-measure)", CIRCUIT, "flips")
    # Stage 2: absolute sampler on a circuit with random measurements + noise.
    ok &= compare("absolute sampler (random + noisy)", ABSOLUTE_CIRCUIT, "absolute")
    print()
    if ok:
        print("PASS: stim-core matches C++ Stim within tolerance")
        return 0
    print("FAIL: rates diverged")
    return 1


if __name__ == "__main__":
    sys.exit(main())
