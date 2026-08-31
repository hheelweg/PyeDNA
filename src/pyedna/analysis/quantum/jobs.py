"""Quantum job scheduling for analysis groups."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
import warnings

from pyedna.analysis.runtime import configure_thread_environment, detect_runtime_resources
from pyedna.analysis.quantum.base import get_quantum_backend


TDDFT_OUTPUT_KEYS = {
    "energies": "exc",
    "excited_state_energies": "exc",
    "excitation_energies": "exc",
    "oscillator_strengths": "osc",
    "transition_dipoles": "dip",
    "transition_quadrupoles": "quad",
    "transition_density_matrices": "tdm",
    "tdm": "tdm",
    "strongest_state": "idx",
    "mulliken": ("mull_pops", "mull_chrgs"),
    "mulliken_populations": "mull_pops",
    "mulliken_charges": "mull_chrgs",
    "opa": "OPA",
    "orbital_participation": "OPA",
}

DEFAULT_TDDFT_OUTPUTS = ["energies", "oscillator_strengths", "transition_dipoles"]


@dataclass(frozen=True)
class MoleculeSummary:
    natm: int
    charge: int
    spin: int


@dataclass(frozen=True)
class QuantumResult:
    frame: int
    group: str
    method: str
    molecule: object
    mean_field: object
    occupied_orbitals: object
    virtual_orbitals: object
    orbital_energies: object
    tddft: dict
    molecule_input: object
    dft_settings: dict
    write_outputs: list


def run_quantum_jobs(config, groups, frame, group_fragments=None, resources=None):
    jobs = config.get("quantum", [])
    if not jobs:
        return []

    _validate_quantum_groups(jobs, groups)

    scheduler = config.get("quantum_scheduler", {})
    resources = detect_runtime_resources() if resources is None else resources
    if scheduler.get("parallel", False):
        return run_quantum_jobs_parallel(
            jobs,
            groups,
            frame,
            scheduler,
            group_fragments or {},
            resources=resources,
        )

    return [
        run_quantum_job(
            job,
            groups[job["group"]],
            frame,
            (group_fragments or {}).get(job["group"], []),
            resources=resources,
        )
        for job in jobs
    ]


def run_quantum_jobs_parallel(jobs, groups, frame, scheduler, group_fragments=None, resources=None):
    resources = detect_runtime_resources() if resources is None else resources
    device = _device_from_resources(resources)
    gpu_ids = _scheduler_gpu_ids(scheduler, resources) if device == "gpu" else []
    max_workers = _scheduler_max_workers(scheduler, resources, device, len(jobs))
    threads_per_worker = _threads_per_worker(resources, max_workers)

    payloads = []
    for index, job in enumerate(jobs):
        group_mol = groups[job["group"]]
        gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else None
        fragments = (group_fragments or {}).get(job["group"], [])
        payloads.append((
            job,
            _group_payload(group_mol, fragments),
            frame,
            device,
            gpu_id,
            threads_per_worker,
        ))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_quantum_payload_worker, payload)
            for payload in payloads
        ]
        return [future.result() for future in futures]


def run_quantum_job(job, group_mol, frame, fragments=None, resources=None):
    resources = detect_runtime_resources() if resources is None else resources

    return _run_quantum_payload(
        job,
        _group_payload(group_mol, fragments or []),
        frame,
        device=_device_from_resources(resources),
    )


def pyscf_mol_to_atom_list(mol):
    return [
        (mol.atom_symbol(i), tuple(mol.atom_coord(i, unit="Angstrom")))
        for i in range(mol.natm)
    ]


def summarize_quantum_result(result):
    parts = [
        f"Frame {result.frame}: quantum {result.method} group {result.group}",
        f"{result.molecule.natm} atoms",
        f"charge {result.molecule.charge}",
    ]

    if result.tddft:
        if "exc" in result.tddft:
            parts.append(f"exc={_format_sequence(result.tddft['exc'])}")
        if "osc" in result.tddft:
            parts.append(f"osc={_format_sequence(result.tddft['osc'])}")
        if "idx" in result.tddft:
            parts.append(f"strongest={result.tddft['idx']}")
        if "mulliken_fragments" in result.tddft:
            parts.append("mulliken_fragments=yes")
        if "OPA" in result.tddft:
            parts.append("opa=yes")

    return ", ".join(parts)


def _run_quantum_payload_worker(payload):
    job, group_payload, frame, device, gpu_id, threads_per_worker = payload
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _set_worker_threads(threads_per_worker)
    return _run_quantum_payload(
        job,
        group_payload,
        frame,
        device=device,
        worker=True,
    )


def _run_quantum_payload(job, group_payload, frame, device="cpu", worker=False):
    _warn_ignored_gpu_setting(job)
    backend = get_quantum_backend(job.get("backend", "pyscf"), device=device)
    method = job["method"]
    molecule_input = group_payload["atoms"]
    dft_settings = _dft_settings(job, group_payload)

    dft_result = backend.run_dft(
        molecule_input,
        molecule_id=job["group"],
        settings=dft_settings,
    )
    mol = dft_result.molecule

    tddft = {}
    if method == "tddft":
        _validate_fragment_outputs(job, group_payload)
        quantum_dict = _tddft_output_flags(job.get("outputs", DEFAULT_TDDFT_OUTPUTS))
        state_ids = _state_ids(job)
        tddft = backend.run_tddft(
            dft_result,
            quantum_dict,
            {
                "fragments": _fragment_indices(group_payload),
                "state_ids": state_ids,
                "tda": job.get("tda", True),
                "singlet": job.get("singlet", True),
            },
        )
        _add_fragment_summaries(tddft, group_payload)

    molecule = mol
    if worker:
        molecule = MoleculeSummary(natm=mol.natm, charge=mol.charge, spin=mol.spin)

    return QuantumResult(
        frame=frame,
        group=job["group"],
        method=method,
        molecule=molecule,
        mean_field=None if worker else dft_result.mean_field,
        occupied_orbitals=None if worker else dft_result.occupied_orbitals,
        virtual_orbitals=None if worker else dft_result.virtual_orbitals,
        orbital_energies=dft_result.orbital_energies,
        tddft=tddft,
        molecule_input=molecule_input,
        dft_settings=dft_settings,
        write_outputs=job.get("_write_outputs", job.get("outputs", [])),
    )


def _group_payload(mol, fragments=None):
    return {
        "atoms": pyscf_mol_to_atom_list(mol),
        "basis": getattr(mol, "basis", "6-31g"),
        "charge": mol.charge,
        "spin": mol.spin,
        "fragments": [
            {
                "name": fragment.name,
                "residue": fragment.residue,
                "atom_indices": fragment.atom_indices,
            }
            for fragment in (fragments or [])
        ],
    }


def _dft_settings(job, group_mol):
    return {
        "basis": job.get("basis", _get_group_value(group_mol, "basis", "6-31g")),
        "xc": job.get("xc", "b3lyp"),
        "density_fit": job.get("density_fit", False),
        "charge": job.get("charge", _get_group_value(group_mol, "charge", 0)),
        "spin": job.get("spin", _get_group_value(group_mol, "spin", 0)),
        "scf_cycles": job.get("scf_cycles", 200),
        "verbosity": job.get("verbosity", 4),
        "optimize_cap": False,
    }


def _get_group_value(group_mol, key, default):
    if isinstance(group_mol, dict):
        return group_mol.get(key, default)
    return getattr(group_mol, key, default)


def _device_from_resources(resources):
    return "gpu" if resources.has_gpu else "cpu"


def _scheduler_gpu_ids(scheduler, resources):
    if "gpu_ids" in scheduler:
        warnings.warn(
            "[quantum_scheduler].gpu_ids is deprecated; GPU assignment is "
            "normally inferred from the scheduler allocation.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [str(gpu_id) for gpu_id in scheduler["gpu_ids"]]
    return list(resources.gpu_ids)


def _scheduler_max_workers(scheduler, resources, device, num_jobs):
    if "max_workers" in scheduler:
        return min(scheduler["max_workers"], num_jobs)
    if device == "gpu":
        return max(1, min(resources.num_gpus, num_jobs))
    return 1


def _threads_per_worker(resources, max_workers):
    return max(1, resources.num_cpus // max(1, max_workers))


def _set_worker_threads(threads):
    configure_thread_environment(threads)


def _warn_ignored_gpu_setting(job):
    if "gpu" not in job:
        return
    warnings.warn(
        "[[quantum]].gpu is deprecated and ignored; PySCF CPU/GPU execution is "
        "selected from resources visible to the process.",
        DeprecationWarning,
        stacklevel=2,
    )


def _validate_quantum_groups(jobs, groups):
    for index, job in enumerate(jobs, start=1):
        group_name = job["group"]
        if group_name not in groups:
            raise ValueError(f"[[quantum]] block {index} references undefined group '{group_name}'")


def _state_ids(job):
    if "state_ids" in job:
        return job["state_ids"]

    nstates = job.get("nstates", 1)
    return list(range(nstates))


def _tddft_output_flags(outputs):
    flags = {
        "exc": False,
        "tdm": False,
        "dip": False,
        "quad": False,
        "osc": False,
        "idx": False,
        "mull_pops": False,
        "mull_chrgs": False,
        "OPA": False,
    }

    for output in outputs:
        keys = TDDFT_OUTPUT_KEYS[output]
        if isinstance(keys, tuple):
            for key in keys:
                flags[key] = True
        else:
            flags[keys] = True

    return flags


def _validate_fragment_outputs(job, group_payload):
    outputs = job.get("outputs", DEFAULT_TDDFT_OUTPUTS)
    if "opa" not in outputs and "orbital_participation" not in outputs:
        return

    fragments = group_payload.get("fragments", [])
    if len(fragments) != 2:
        raise ValueError(
            f"OPA for group '{job['group']}' requires exactly two fragments. "
            "Use a [[groups]] block with exactly two attachment residues."
        )


def _fragment_indices(group_payload):
    fragments = group_payload.get("fragments", [])
    if not fragments:
        return None
    return [fragment["atom_indices"] for fragment in fragments]


def _add_fragment_summaries(tddft, group_payload):
    if "mull_pops" not in tddft and "mull_chrgs" not in tddft:
        return

    fragments = group_payload.get("fragments", [])
    if not fragments:
        return

    nstates = len(tddft.get("mull_pops", tddft.get("mull_chrgs", [])))
    summaries = []

    for state_index in range(nstates):
        state_summary = []
        for fragment in fragments:
            indices = fragment["atom_indices"]
            entry = {
                "name": fragment["name"],
                "residue": fragment["residue"],
            }
            if "mull_pops" in tddft:
                atom_pops = tddft["mull_pops"][state_index]
                entry["population"] = float(sum(abs(atom_pops[index]) for index in indices))
            if "mull_chrgs" in tddft:
                atom_charges = tddft["mull_chrgs"][state_index]
                entry["charge"] = float(sum(atom_charges[index] for index in indices))
            state_summary.append(entry)
        summaries.append(state_summary)

    tddft["mulliken_fragments"] = summaries


def _format_sequence(values):
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"
