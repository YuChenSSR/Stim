//! PyO3 bindings exposing `stim-core` to Python as the `stimcore` module.
//!
//! This is the proof-of-concept's "product" layer: it gives the Rust simulators
//! a small, Stim-flavoured Python API so they can be driven from notebooks and
//! cross-checked directly against the `stim` pip package.
//!
//! ```python
//! import stimcore
//! c = stimcore.Circuit("H 0\nCX 0 1\nM 0 1")
//! c.num_qubits            # -> 2
//! c.sample(shots=1000)    # -> numpy bool array (1000, 2)
//! det, obs = stimcore.Circuit(dem_text).sample_detectors(shots=1000)
//! ```

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use stim_core::Circuit as CoreCircuit;

/// Builds a `(shots, cols)` numpy bool array from a row-major bit matrix.
fn to_array2<'py>(
    py: Python<'py>,
    rows: &[Vec<bool>],
    cols: usize,
) -> Bound<'py, PyArray2<bool>> {
    let shots = rows.len();
    let mut arr = Array2::<bool>::default((shots, cols));
    for (i, row) in rows.iter().enumerate() {
        for (j, &b) in row.iter().enumerate() {
            arr[[i, j]] = b;
        }
    }
    arr.into_pyarray_bound(py)
}

/// A quantum stabilizer circuit (subset of Stim's `.stim` format).
#[pyclass]
struct Circuit {
    inner: CoreCircuit,
}

#[pymethods]
impl Circuit {
    /// Parses a circuit from Stim text format (supported subset).
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        CoreCircuit::from_text(text)
            .map(|inner| Circuit { inner })
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn num_measurements(&self) -> usize {
        self.inner.num_measurements()
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Samples absolute measurements: a `(shots, num_measurements)` bool array.
    /// Analogue of `stim.Circuit.compile_sampler().sample(shots)`.
    #[pyo3(signature = (shots, seed = 0))]
    fn sample<'py>(&self, py: Python<'py>, shots: usize, seed: u64) -> Bound<'py, PyArray2<bool>> {
        let samples = stim_core::sample(&self.inner, shots, seed);
        to_array2(py, &samples, self.inner.num_measurements())
    }

    /// Samples detection events and observable flips: a pair of bool arrays
    /// `((shots, num_detectors), (shots, num_observables))`. Analogue of
    /// `stim.Circuit.compile_detector_sampler().sample(shots,
    /// separate_observables=True)`.
    #[pyo3(signature = (shots, seed = 0))]
    fn sample_detectors<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
    ) -> (Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>) {
        let s = stim_core::sample_detectors(&self.inner, shots, seed);
        let det = to_array2(py, &s.detectors, self.inner.num_detectors());
        let obs = to_array2(py, &s.observables, self.inner.num_observables());
        (det, obs)
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(num_qubits={}, num_measurements={}, num_detectors={})",
            self.inner.num_qubits(),
            self.inner.num_measurements(),
            self.inner.num_detectors(),
        )
    }
}

#[pymodule]
fn stimcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust proof-of-concept port of Stim (core sampling).")?;
    m.add_class::<Circuit>()?;
    Ok(())
}
