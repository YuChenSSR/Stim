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

ROOT = pathlib.Path(__file__).resolve().parents[3]
CRATE = ROOT / "rust"


def stim_rates():
    circuit = stim.Circuit(CIRCUIT)
    sampler = circuit.compile_sampler()
    shots = sampler.sample(SHOTS)
    return shots.mean(axis=0)


def rust_rates():
    # A tiny Rust harness compiled on the fly via `cargo run --example`.
    example = CRATE / "stim-core" / "examples" / "rates.rs"
    example.parent.mkdir(parents=True, exist_ok=True)
    example.write_text(
        '''
use stim_core::{measurement_flip_rates, Circuit};
fn main() {
    let text = std::env::args().nth(1).unwrap();
    let shots: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let c = Circuit::from_text(&text).unwrap();
    let rates = measurement_flip_rates(&c, shots, 1234);
    let parts: Vec<String> = rates.iter().map(|r| format!("{r}")).collect();
    println!("{}", parts.join(","));
}
'''
    )
    out = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "rates", "--",
         CIRCUIT, str(SHOTS)],
        cwd=CRATE,
        capture_output=True,
        text=True,
        check=True,
    )
    line = out.stdout.strip().splitlines()[-1]
    return np.array([float(x) for x in line.split(",")])


def main():
    s = stim_rates()
    r = rust_rates()
    print(f"stim  rates: {np.array2string(s, precision=4)}")
    print(f"rust  rates: {np.array2string(r, precision=4)}")
    assert s.shape == r.shape, f"shape mismatch {s.shape} vs {r.shape}"
    diff = np.abs(s - r)
    print(f"max abs diff: {diff.max():.4f} (tol {TOL})")
    if diff.max() < TOL:
        print("PASS: Rust flip rates match C++ Stim within tolerance")
        return 0
    print("FAIL: rates diverged")
    return 1


if __name__ == "__main__":
    sys.exit(main())
