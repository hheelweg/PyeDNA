"""Structured output writers for trajectory analysis."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil

from .units import DEFAULT_ANALYSIS_UNITS, distance_factor, energy_factor, merged_units

DEFAULT_QUANTUM_OUTPUT = "quantum.jsonl"
DEFAULT_QUANTUM_INTERACTIONS_OUTPUT = "quantum_interactions.jsonl"
DEFAULT_CLASSICAL_INTERACTIONS_OUTPUT = "classical_interactions.jsonl"
DEFAULT_CLASSICAL_OUTPUT = "classical.jsonl"


@dataclass(frozen=True)
class LoadedAnalysisRun:
    directory: Path
    manifest: dict
    quantum: list
    classical: list
    quantum_interactions: list
    classical_interactions: list

    def quantum_dataframe(self):
        return records_dataframe(self.quantum)

    def classical_dataframe(self):
        return records_dataframe(self.classical)

    def quantum_interactions_dataframe(self):
        return records_dataframe(self.quantum_interactions)

    def classical_interactions_dataframe(self):
        return records_dataframe(self.classical_interactions)


@dataclass(frozen=True)
class AnalysisRun:
    directory: Path
    quantum_file: Path
    quantum_interactions_file: Path
    classical_interactions_file: Path
    classical_file: Path
    manifest_file: Path
    config_file: Path
    units: dict


def prepare_output_files(config, config_file=None):
    run = create_analysis_run(config, config_file=config_file)

    for path in (
        run.quantum_file,
        run.quantum_interactions_file,
        run.classical_interactions_file,
        run.classical_file,
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()

    return run


def create_analysis_run(config, config_file=None):
    analysis = config.get("analysis", {})
    output_root = Path(analysis.get("output_root", "analysis"))
    name = analysis.get("name", "auto")
    directory = _analysis_directory(output_root, name)
    directory.mkdir(parents=True, exist_ok=False)

    copied_config = directory / "traj.toml"
    if config_file is not None:
        shutil.copy2(config_file, copied_config)

    output = config.get("output", {})
    run = AnalysisRun(
        directory=directory,
        quantum_file=directory / output.get("quantum_file", DEFAULT_QUANTUM_OUTPUT),
        quantum_interactions_file=directory / output.get(
            "quantum_interactions_file",
            output.get("interaction_file", DEFAULT_QUANTUM_INTERACTIONS_OUTPUT),
        ),
        classical_interactions_file=directory / output.get(
            "classical_interactions_file",
            DEFAULT_CLASSICAL_INTERACTIONS_OUTPUT,
        ),
        classical_file=directory / output.get("classical_file", DEFAULT_CLASSICAL_OUTPUT),
        manifest_file=directory / "manifest.json",
        config_file=copied_config,
        units=analysis.get("units", DEFAULT_ANALYSIS_UNITS),
    )
    write_manifest(config, run)
    return run


def write_manifest(config, run):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "directory": str(run.directory),
        "config_file": str(run.config_file),
        "trajectory": config.get("trajectory", {}),
        "units": config.get("analysis", {}).get("units", DEFAULT_ANALYSIS_UNITS),
        "outputs": {
            "quantum": str(run.quantum_file),
            "classical": str(run.classical_file),
            "quantum_interactions": str(run.quantum_interactions_file),
            "classical_interactions": str(run.classical_interactions_file),
        },
        "classical_interactions": config.get("classical_interactions", []),
        "quantum_interactions": config.get("quantum_interactions", []),
        "quantum_jobs": [
            {
                "group": job.get("group"),
                "method": job.get("method"),
                "write_outputs": job.get("_write_outputs", job.get("outputs", [])),
                "compute_outputs": job.get("_compute_outputs", job.get("outputs", [])),
            }
            for job in config.get("quantum", [])
        ],
    }

    with run.manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)


def quantum_output_file(target):
    return _output_path(target, "quantum_file", DEFAULT_QUANTUM_OUTPUT)


def quantum_interactions_output_file(target):
    return _output_path(target, "quantum_interactions_file", DEFAULT_QUANTUM_INTERACTIONS_OUTPUT)


def classical_interactions_output_file(target):
    return _output_path(target, "classical_interactions_file", DEFAULT_CLASSICAL_INTERACTIONS_OUTPUT)


def interaction_output_file(target):
    return quantum_interactions_output_file(target)


def classical_output_file(target):
    return _output_path(target, "classical_file", DEFAULT_CLASSICAL_OUTPUT)


def append_quantum_results(target, results):
    path = quantum_output_file(target)
    if path is None or not results:
        return

    with path.open("a") as f:
        for result in results:
            f.write(json.dumps(quantum_result_record(result, units=_analysis_units(target))) + "\n")


def append_interaction_results(target, results):
    append_quantum_interaction_results(target, [result for result in results if result.type == "coupling"])
    append_classical_interaction_results(target, [result for result in results if result.type == "distance"])


def append_quantum_interaction_results(target, results):
    path = quantum_interactions_output_file(target)
    if path is None or not results:
        return

    with path.open("a") as f:
        for result in results:
            f.write(json.dumps(quantum_interaction_result_record(result, units=_analysis_units(target))) + "\n")


def append_classical_interaction_results(target, results):
    path = classical_interactions_output_file(target)
    if path is None or not results:
        return

    with path.open("a") as f:
        for result in results:
            f.write(json.dumps(classical_interaction_result_record(result, units=_analysis_units(target))) + "\n")


def append_classical_results(target, results):
    path = classical_output_file(target)
    if path is None or not results:
        return

    with path.open("a") as f:
        for result in results:
            f.write(json.dumps(classical_result_record(result, units=_analysis_units(target))) + "\n")


def quantum_result_record(result, units=None):
    record = {
        "frame": result.frame,
        "group": result.group,
        "method": result.method,
        "atom_count": result.molecule.natm,
        "charge": result.molecule.charge,
        "spin": result.molecule.spin,
    }

    if result.tddft:
        record["tddft"] = _filtered_tddft_outputs(result, units=merged_units(units))

    return record


def interaction_result_record(result, units=None):
    if result.type == "distance":
        return classical_interaction_result_record(result, units=units)
    return quantum_interaction_result_record(result, units=units)


def quantum_interaction_result_record(result, units=None):
    return {
        "frame": result.frame,
        "type": result.type,
        "method": result.method,
        "groups": result.groups,
        "state_pair": result.state_pair,
        "values": _convert_quantum_interaction_values(result, merged_units(units)),
    }


def classical_interaction_result_record(result, units=None):
    return {
        "frame": result.frame,
        "type": result.type,
        "method": result.method,
        "groups": result.groups,
        "values": _convert_classical_interaction_values(result, merged_units(units)),
    }


def classical_result_record(result, units=None):
    return {
        "frame": result.frame,
        "group": result.group,
        "values": _convert_classical_values(result.values, merged_units(units)),
    }


def _to_json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "item"):
        return _to_json_value(value.item())
    return value


def _analysis_directory(output_root, name):
    if name == "auto":
        base = output_root / f"analysis_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    else:
        base = output_root / name

    if not base.exists():
        return base

    for index in range(1, 100):
        candidate = Path(f"{base}_{index:02d}")
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"Could not find available analysis directory based on {base}")


def _output_path(target, attr, default):
    if isinstance(target, AnalysisRun):
        return getattr(target, attr)

    output = target.get("output", {})
    if attr == "quantum_interactions_file":
        return Path(output.get(attr, output.get("interaction_file", default)))
    return Path(output.get(attr, default))


def _analysis_units(target):
    if isinstance(target, AnalysisRun):
        return target.units
    if isinstance(target, dict):
        return target.get("analysis", {}).get("units", DEFAULT_ANALYSIS_UNITS)
    return DEFAULT_ANALYSIS_UNITS


def load_analysis_run(path):
    directory = Path(path)
    manifest_file = directory / "manifest.json"
    with manifest_file.open() as f:
        manifest = json.load(f)

    outputs = manifest.get("outputs", {})
    return LoadedAnalysisRun(
        directory=directory,
        manifest=manifest,
        quantum=read_jsonl(_analysis_output_path(directory, outputs, "quantum")),
        classical=read_jsonl(_analysis_output_path(directory, outputs, "classical")),
        quantum_interactions=read_jsonl(_analysis_output_path(directory, outputs, "quantum_interactions")),
        classical_interactions=read_jsonl(_analysis_output_path(directory, outputs, "classical_interactions")),
    )


def read_jsonl(path):
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Analysis output file does not exist: {path}")

    records = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSONL record {line_number} in {path}") from exc
    return records


def records_dataframe(records):
    import pandas as pd

    rows = [_flatten_record(record) for record in records]
    return pd.DataFrame(rows)


def _analysis_output_path(directory, outputs, key):
    value = outputs.get(key)
    if not value:
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return directory / path


def _flatten_record(record):
    row = {}
    for key, value in record.items():
        _flatten_value(row, key, value)
    return row


def _flatten_value(row, prefix, value):
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_value(row, f"{prefix}.{key}", item)
        return

    if isinstance(value, list):
        _flatten_list(row, prefix, value)
        return

    row[prefix] = value


def _flatten_list(row, prefix, value):
    if _is_scalar_list(value):
        for index, item in enumerate(value):
            row[f"{prefix}.{index}"] = item
        return

    row[prefix] = value


def _is_scalar_list(value):
    return all(not isinstance(item, (dict, list)) for item in value)


def _filtered_tddft_outputs(result, units=None):
    requested = result.write_outputs or list(result.tddft)
    output = {}

    for name in requested:
        for key in _output_keys_for_name(name):
            if key in result.tddft:
                public_name = _public_output_name(key, name)
                value = _convert_quantum_output(public_name, result.tddft[key], units)
                output[public_name] = _to_json_value(value)

    return output


def _convert_quantum_output(name, value, units):
    if name == "excited_state_energies":
        return _scale_value(value, energy_factor(units["energy"]))
    return value


def _convert_quantum_interaction_values(result, units):
    values = _to_json_value(result.values)
    if result.type != "coupling":
        return values
    return _scale_matching_keys(values, ("coupling",), energy_factor(units["coupling"]))


def _convert_classical_interaction_values(result, units):
    values = _to_json_value(result.values)
    if result.type != "distance":
        return values
    return _scale_matching_keys(values, ("distance",), distance_factor(units["distance"]))


def _convert_classical_values(values, units):
    values = _to_json_value(values)
    return _scale_matching_keys(
        values,
        ("center_of_geometry", "center_of_mass", "radius_of_gyration"),
        distance_factor(units["distance"]),
    )


def _scale_matching_keys(value, names, factor):
    if isinstance(value, dict):
        return {
            key: _scale_value(item, factor) if _matches_named_quantity(key, names) else _scale_matching_keys(item, names, factor)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scale_matching_keys(item, names, factor) for item in value]
    return value


def _matches_named_quantity(key, names):
    return any(key == name or key.startswith(f"{name} ") for name in names)


def _scale_value(value, factor):
    value = _to_json_value(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value * factor
    if isinstance(value, list):
        return [_scale_value(item, factor) for item in value]
    if isinstance(value, dict):
        return {key: _scale_value(item, factor) for key, item in value.items()}
    return value


def _output_keys_for_name(name):
    mapping = {
        "energies": ("exc",),
        "excited_state_energies": ("exc",),
        "excitation_energies": ("exc",),
        "oscillator_strengths": ("osc",),
        "strongest_state": ("idx",),
        "transition_dipoles": ("dip",),
        "transition_quadrupoles": ("quad",),
        "tdm": ("tdm",),
        "transition_density_matrices": ("tdm",),
        "mulliken": ("mulliken_fragments",),
        "mulliken_populations": ("mull_pops", "mulliken_fragments"),
        "mulliken_charges": ("mull_chrgs", "mulliken_fragments"),
        "opa": ("OPA",),
        "orbital_participation": ("OPA",),
    }
    return mapping.get(name, (name,))


def _public_output_name(key, requested_name):
    if key == "exc":
        return "excited_state_energies"
    if key == "osc":
        return "oscillator_strengths"
    if key == "idx":
        return "strongest_state"
    if key == "OPA":
        return "orbital_participation"
    return key
