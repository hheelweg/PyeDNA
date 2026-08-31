"""Quantum backend interface and registry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DFTResult:
    molecule: object
    mean_field: object
    occupied_orbitals: object
    virtual_orbitals: object
    orbital_energies: object


class QuantumBackend:
    name = None

    def run_dft(self, molecule_input, molecule_id, settings):
        raise NotImplementedError

    def run_tddft(self, dft_result, output_flags, settings):
        raise NotImplementedError


def get_quantum_backend(name="pyscf", device="cpu"):
    name = (name or "pyscf").lower()

    if name == "pyscf":
        from .pyscf import PySCFBackend

        return PySCFBackend(device=device)

    if name == "orca":
        from .orca import ORCABackend

        return ORCABackend()

    raise ValueError(f"Unsupported quantum backend '{name}'")
