"""Topology-based Amber positional restraint selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import parmed as pmd


DNA_RESIDUES = {
    "DA",
    "DC",
    "DG",
    "DT",
    "DA5",
    "DC5",
    "DG5",
    "DT5",
    "DA3",
    "DC3",
    "DG3",
    "DT3",
}
TERMINAL_DNA_RESIDUES = {"DA5", "DC5", "DG5", "DT5", "DA3", "DC3", "DG3", "DT3"}
INTERNAL_STAGES = ("min1", "min2", "eq1", "eq2", "prod")


@dataclass(frozen=True)
class ResidueInfo:
    """Store a residue number and name."""

    number: int
    name: str

    def label(self):
        return f"{self.number} {self.name}"


@dataclass(frozen=True)
class ResolvedRestraint:
    """Store a resolved Amber restraint for one internal MD stage."""

    target: str
    strength: Optional[float] = None
    residues: tuple[int, ...] = ()

    @property
    def active(self):
        return self.target != "none"

    @property
    def mask(self):
        if not self.active:
            return None
        return ":" + ",".join(_format_ranges(self.residues))

    @property
    def ranges(self):
        return tuple(_collapse_ranges(self.residues))


class AmberRestraintResolver:
    """Resolve md.toml restraint targets from an Amber topology."""

    def __init__(self, prmtop_file, config):
        self.config = config
        self.parm = pmd.load_file(str(prmtop_file))
        self.residues = [
            ResidueInfo(index, residue.name.strip())
            for index, residue in enumerate(self.parm.residues, start=1)
        ]
        self._molecule_ranges = _molecule_ranges(
            self.parm.parm_data.get("ATOMS_PER_MOLECULE", [])
        )
        self._first_solvent_molecule = _first_solvent_molecule(
            self.parm.parm_data.get("SOLVENT_POINTERS", [])
        )
        self._atom_to_molecule = self._build_atom_to_molecule()
        self._bonded_atoms = {
            atom.idx
            for bond in self.parm.bonds
            for atom in (bond.atom1, bond.atom2)
        }
        self._cache = {}
        self._analysis = None

    def for_stage(self, stage):
        """Return the resolved restraint for an internal Amber stage."""

        if stage not in self._cache:
            stage_config = self._stage_config(stage)
            residues = self._target_residues(stage_config.target)
            self._cache[stage] = ResolvedRestraint(
                target=stage_config.target,
                strength=stage_config.strength,
                residues=tuple(residue.number for residue in residues),
            )
        return self._cache[stage]

    def analysis_text(self):
        """Return a human-readable restraint analysis for the MD log."""

        analysis = self._topology_analysis()
        lines = [
            "Restraint analysis",
            "------------------",
            "Terminal DNA residues:",
            *self._format_residue_lines(analysis["terminal"]),
            "",
            "Structure residues:",
            "  DNA residues:",
            *self._format_residue_lines(analysis["structure_dna"], indent="    "),
            "  custom residues:",
            *self._format_residue_lines(analysis["structure_custom"], indent="    "),
            "",
            "Excluded:",
            "  solvent residues:",
            *self._format_residue_lines(analysis["solvent"], indent="    "),
            "  ion residues:",
            *self._format_residue_lines(analysis["ions"], indent="    "),
            "",
            "Stage restraints:",
        ]

        for stage in INTERNAL_STAGES:
            restraint = self.for_stage(stage)
            lines.extend([
                f"  {stage}:",
                f"      target = {restraint.target}",
                f"      strength = {restraint.strength}",
                f"      mask = {restraint.mask}",
                "",
            ])
        return "\n".join(lines).rstrip()

    def _stage_config(self, stage):
        if stage == "min1":
            return self.config.minimization.restraints.stage1
        if stage == "min2":
            return self.config.minimization.restraints.stage2
        if stage == "eq1":
            return self.config.equilibration.restraints.stage1
        if stage == "eq2":
            return self.config.equilibration.restraints.stage2
        if stage == "prod":
            return self.config.production.restraints
        raise ValueError(f"Unknown MD stage: {stage}")

    def _target_residues(self, target):
        if target == "none":
            return ()
        if target == "custom":
            raise NotImplementedError("Custom restraint targets are not implemented yet")

        analysis = self._topology_analysis()
        if target == "terminal":
            residues = analysis["terminal"]
        elif target == "structure":
            residues = analysis["structure"]
        else:
            raise ValueError(f"Unknown restraint target: {target}")

        if not residues:
            raise ValueError(f"No residues found for restraint target {target!r}")
        return residues

    def _topology_analysis(self):
        if self._analysis is None:
            terminal = [
                residue for residue in self.residues
                if residue.name in TERMINAL_DNA_RESIDUES
            ]
            solvent = self._solvent_residues()
            ions = self._isolated_ion_residues()
            excluded = {residue.number for residue in solvent + ions}
            structure = [
                residue for residue in self.residues
                if residue.number not in excluded
            ]
            structure_dna = [
                residue for residue in structure
                if residue.name in DNA_RESIDUES
            ]
            structure_custom = [
                residue for residue in structure
                if residue.name not in DNA_RESIDUES
            ]

            self._analysis = {
                "terminal": terminal,
                "structure": structure,
                "structure_dna": structure_dna,
                "structure_custom": structure_custom,
                "solvent": solvent,
                "ions": ions,
            }
        return self._analysis

    def _solvent_residues(self):
        first_solvent = self._first_solvent_molecule
        if first_solvent is None:
            return []

        return [
            residue for residue in self.residues
            if self._residue_molecules(residue.number)
            and min(self._residue_molecules(residue.number)) >= first_solvent
        ]

    def _isolated_ion_residues(self):
        return [
            residue for residue in self.residues
            if self._is_single_unbonded_atom_residue(residue.number)
        ]

    def _is_single_unbonded_atom_residue(self, residue_number):
        residue = self.parm.residues[residue_number - 1]
        return len(residue.atoms) == 1 and residue.atoms[0].idx not in self._bonded_atoms

    def _residue_molecules(self, residue_number):
        return {
            self._atom_to_molecule[atom.idx]
            for atom in self.parm.residues[residue_number - 1].atoms
            if atom.idx in self._atom_to_molecule
        }

    def _build_atom_to_molecule(self):
        atom_to_molecule = {}
        for molecule_index, (start, end) in enumerate(self._molecule_ranges, start=1):
            for atom_index in range(start, end):
                atom_to_molecule[atom_index] = molecule_index
        return atom_to_molecule

    @staticmethod
    def _format_residue_lines(residues, indent="  "):
        if not residues:
            return [f"{indent}(none)"]
        return [f"{indent}{residue.label()}" for residue in residues]


def _first_solvent_molecule(solvent_pointers):
    if len(solvent_pointers) < 3:
        return None
    first = solvent_pointers[2]
    return first if first > 0 else None


def _molecule_ranges(atoms_per_molecule):
    ranges = []
    start = 0
    for atom_count in atoms_per_molecule:
        end = start + atom_count
        ranges.append((start, end))
        start = end
    return ranges


def _format_ranges(values):
    return [
        f"{start}-{end}" if start != end else str(start)
        for start, end in _collapse_ranges(values)
    ]


def _collapse_ranges(values):
    values = sorted(set(values))
    if not values:
        return []

    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return ranges
