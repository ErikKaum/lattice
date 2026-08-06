//! PyO3 bindings — feature-gated by `python`.
//!
//! Surface mirrors the Rust API. Two classes:
//!
//! - `Model`: load from a `.safetensors` path, embed a token-id sequence,
//!   read metadata accessors (`dim`, `vocab`, `variant`, `bits`, `axis`).
//! - `Tokenizer`: load a `tokenizer.json`, encode a single text or a batch.
//!
//! Embeddings return as `numpy.ndarray[float32]`, dim-shaped. The bindings
//! own an `EmbedScratch` per `Model` so per-call work doesn't allocate.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::{Axis, EmbedScratch, LatticeTokenizer, Model, kernel};

/// Map an `anyhow::Error` into a Python `RuntimeError`. We expose everything
/// as `RuntimeError` for now; the Rust side already distinguishes
/// IO/parse/validation errors via the message, and the Python test layer
/// doesn't branch on type.
fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pyclass(name = "Model", unsendable)]
struct PyModel {
    inner: Model,
    scratch: EmbedScratch,
}

#[pymethods]
impl PyModel {
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let model = Model::load(&PathBuf::from(path)).map_err(err)?;
        let scratch = model.scratch();
        Ok(Self {
            inner: model,
            scratch,
        })
    }

    /// Embed one token-id sequence. Returns a 1-D float32 ndarray of length
    /// `dim`. Empty input → zero vector (matches `EmbeddingBag` semantics).
    ///
    /// If `normalize=True` (default), the output is L2-normalized — same as
    /// what training and eval consumed. Pass `normalize=False` to inspect
    /// raw mean-pool magnitudes.
    #[pyo3(signature = (tokens, normalize = true))]
    fn embed<'py>(
        &mut self,
        py: Python<'py>,
        tokens: Vec<u32>,
        normalize: bool,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let dim = self.inner.dim();
        let mut out = vec![0.0f32; dim];
        self.inner
            .embed(&tokens, &mut out, &mut self.scratch)
            .map_err(err)?;
        if normalize {
            kernel::l2_normalize(&mut out);
        }
        Ok(out.into_pyarray_bound(py))
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    #[getter]
    fn vocab(&self) -> usize {
        self.inner.vocab()
    }
    #[getter]
    fn variant(&self) -> String {
        self.inner.variant().to_string()
    }
    #[getter]
    fn bits(&self) -> u8 {
        self.inner.bits()
    }
    #[getter]
    fn axis(&self) -> &'static str {
        match self.inner.axis() {
            Axis::Dim => "dim",
            Axis::Row => "row",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<lattice.Model variant={} dim={} vocab={}>",
            self.inner.variant(),
            self.inner.dim(),
            self.inner.vocab(),
        )
    }
}

#[pyclass(name = "Tokenizer", unsendable)]
struct PyTokenizer {
    inner: LatticeTokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let inner = LatticeTokenizer::load(&PathBuf::from(path)).map_err(err)?;
        Ok(Self { inner })
    }

    /// Encode one document without automatically adding `[CLS]`/`[SEP]`.
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.encode(text).map_err(err)
    }

    /// Encode a batch of documents. Returns one list per input.
    fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        self.inner.encode_batch(texts).map_err(err)
    }
}

/// The `lattice` Python module. Maturin builds this into a `.so`/`.pyd`
/// when the `python` feature is on.
#[pymodule]
fn lattice(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
