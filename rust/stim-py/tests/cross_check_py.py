#!/usr/bin/env python3
"""Validate the `stimcore` Python extension against the C++ `stim` package.

Builds the extension via cargo, imports it, and compares per-column sampling
rates (measurements, detectors, observables) against stim to sampling noise.

Run from the repo root after `pip install stim`:
    python3 rust/stim-py/tests/cross_check_py.py
"""
import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile

try:
    import stim
    import numpy as np
except ImportError as e:
    print(f"SKIP: missing dependency ({e}); need `pip install stim numpy`")
    sys.exit(0)

SHOTS = 200_000
TOL = 0.01

ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "rust"


def build_and_import():
    subprocess.run(
        ["cargo", "build", "--release", "-p", "stim-py"],
        cwd=WORKSPACE,
        check=True,
    )
    so = WORKSPACE / "target" / "release" / "libstimcore.so"
    tmp = pathlib.Path(tempfile.mkdtemp())
    dst = tmp / "stimcore.so"
    shutil.copy(so, dst)
    spec = importlib.util.spec_from_file_location("stimcore", dst)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(name, stim_rates, rust_rates):
    s = np.asarray(stim_rates, dtype=float)
    r = np.asarray(rust_rates, dtype=float)
    print(f"\n[{name}]")
    print(f"  stim: {np.array2string(s, precision=4)}")
    print(f"  rust: {np.array2string(r, precision=4)}")
    if s.shape != r.shape:
        print(f"  FAIL: shape {s.shape} vs {r.shape}")
        return False
    diff = float(np.abs(s - r).max()) if s.size else 0.0
    ok = diff < TOL
    print(f"  max abs diff: {diff:.4f} (tol {TOL}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    stimcore = build_and_import()
    ok = True

    # 1) Measurement sampling on a GHZ + noise circuit.
    text = "H 0\nCX 0 1\nCX 0 2\nX_ERROR(0.1) 1\nM 0 1 2"
    rust = stimcore.Circuit(text).sample(shots=SHOTS, seed=1).mean(axis=0)
    sref = stim.Circuit(text).compile_sampler().sample(SHOTS).mean(axis=0)
    ok &= check("measurement sampling (GHZ + noise)", sref, rust)

    # 2) Detector + observable sampling on a generated repetition_code circuit.
    gen = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.03,
        before_measure_flip_probability=0.02,
    )
    gen_text = str(gen)
    det_r, obs_r = stimcore.Circuit(gen_text).sample_detectors(shots=SHOTS, seed=1)
    det_s, obs_s = gen.compile_detector_sampler().sample(SHOTS, separate_observables=True)
    ok &= check("detector sampling (repetition_code d3/r3)", det_s.mean(axis=0), det_r.mean(axis=0))
    ok &= check("observable flips (repetition_code d3/r3)", obs_s.mean(axis=0), obs_r.mean(axis=0))

    print()
    if ok:
        print("PASS: stimcore Python extension matches C++ Stim within tolerance")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
