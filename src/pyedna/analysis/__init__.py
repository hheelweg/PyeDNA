"""Analysis workflow, configuration, calculation, and result I/O helpers.

``pyedna.analysis`` owns trajectory-analysis orchestration and computed
quantities. Heavy quantum backends are imported lazily.
"""

import importlib

_LAZY_ATTRS = {
    "AnalysisConfig": "pyedna.analysis.config",
    "AnalysisRun": "pyedna.analysis.io",
    "ClassicalResult": "pyedna.analysis.classical",
    "InteractionResult": "pyedna.analysis.interactions",
    "LoadedAnalysisRun": "pyedna.analysis.io",
    "MoleculeSummary": "pyedna.analysis.quantum.jobs",
    "QuantumResult": "pyedna.analysis.quantum.jobs",
    "RuntimeResources": "pyedna.analysis.runtime",
    "analyze_frame": "pyedna.analysis.workflow",
    "append_classical_interaction_results": "pyedna.analysis.io",
    "append_classical_results": "pyedna.analysis.io",
    "append_interaction_results": "pyedna.analysis.io",
    "append_quantum_interaction_results": "pyedna.analysis.io",
    "append_quantum_results": "pyedna.analysis.io",
    "load_analysis_run": "pyedna.analysis.io",
    "prepare_output_files": "pyedna.analysis.io",
    "read_jsonl": "pyedna.analysis.io",
    "records_dataframe": "pyedna.analysis.io",
    "configure_thread_environment": "pyedna.analysis.runtime",
    "detect_runtime_resources": "pyedna.analysis.runtime",
    "run_classical_interactions": "pyedna.analysis.interactions",
    "run_classical_jobs": "pyedna.analysis.classical",
    "run_interactions": "pyedna.analysis.interactions",
    "run_quantum_interactions": "pyedna.analysis.interactions",
    "run_quantum_job": "pyedna.analysis.quantum.jobs",
    "run_quantum_jobs": "pyedna.analysis.quantum.jobs",
    "run_quantum_jobs_parallel": "pyedna.analysis.quantum.jobs",
    "run_trajectory_analysis": "pyedna.analysis.workflow",
    "summarize_classical_result": "pyedna.analysis.classical",
    "summarize_interaction_result": "pyedna.analysis.interactions",
    "summarize_quantum_result": "pyedna.analysis.quantum.jobs",
    "validate_analysis_config": "pyedna.analysis.config",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
