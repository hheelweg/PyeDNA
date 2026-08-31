"""Quantum-analysis backends and coupling helpers."""

import importlib

_LAZY_ATTRS = {
    "DFTResult": "pyedna.analysis.quantum.base",
    "MoleculeSummary": "pyedna.analysis.quantum.jobs",
    "ORCABackend": "pyedna.analysis.quantum.orca",
    "PySCFBackend": "pyedna.analysis.quantum.pyscf",
    "QuantumBackend": "pyedna.analysis.quantum.base",
    "QuantumResult": "pyedna.analysis.quantum.jobs",
    "RuntimeResources": "pyedna.analysis.runtime",
    "configure_thread_environment": "pyedna.analysis.runtime",
    "detect_runtime_resources": "pyedna.analysis.runtime",
    "get_quantum_backend": "pyedna.analysis.quantum.base",
    "run_dft": "pyedna.analysis.quantum.pyscf",
    "rebuild_pyscf_mol": "pyedna.analysis.quantum.couplings",
    "run_dft_gpu": "pyedna.analysis.quantum.pyscf",
    "run_quantum_job": "pyedna.analysis.quantum.jobs",
    "run_quantum_jobs": "pyedna.analysis.quantum.jobs",
    "run_quantum_jobs_parallel": "pyedna.analysis.quantum.jobs",
    "run_tddft": "pyedna.analysis.quantum.pyscf",
    "run_tddft_gpu": "pyedna.analysis.quantum.pyscf",
    "summarize_quantum_result": "pyedna.analysis.quantum.jobs",
    "tdm_coupling": "pyedna.analysis.quantum.couplings",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
