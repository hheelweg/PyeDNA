"""ORCA quantum-analysis backend.

TODO: Implement ORCA input generation, execution, and output parsing once the
PySCF-backed trajectory workflow is stable.
"""

from .base import QuantumBackend


class ORCABackend(QuantumBackend):
    name = "orca"

    def run_dft(self, molecule_input, molecule_id, settings):
        raise NotImplementedError(
            "The ORCA backend is a TODO. Future work should generate ORCA inputs, "
            "run ORCA, and return a DFTResult-compatible object."
        )

    def run_tddft(self, dft_result, output_flags, settings):
        raise NotImplementedError(
            "The ORCA backend is a TODO. Future work should parse ORCA TDDFT "
            "outputs into the same keys used by the PySCF backend."
        )
