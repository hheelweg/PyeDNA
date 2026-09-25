"""Structured output writers for trajectory analysis."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil

from .units import DEFAULT_ANALYSIS_UNITS, distance_factor, energy_factor, merged_units
from .serialization import (
    ANALYSIS_FORMAT_VERSION,
    RESULT_FAMILIES,
    active_result_families,
    build_result_schemas,
    compatible_array,
    family_row,
    runtime_quantum_metadata,
    schema_metadata_to_manifest,
    schema_has_data_columns,
    schemas_to_manifest,
    validate_row_length,
)

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

    def quantum_dataframe(self, flatten=False):
        return _loaded_dataframe(self, "quantum", flatten=flatten)

    def classical_dataframe(self, flatten=False):
        return _loaded_dataframe(self, "classical", flatten=flatten)

    def quantum_interactions_dataframe(self, flatten=False):
        return _loaded_dataframe(self, "quantum_interactions", flatten=flatten)

    def classical_interactions_dataframe(self, flatten=False):
        return _loaded_dataframe(self, "classical_interactions", flatten=flatten)

    def dataframe(self, family, flatten=False):
        return _loaded_dataframe(self, family, flatten=flatten)

    def array(self, family, column):
        values = []
        for record in _family_records(self, family):
            if column not in record:
                raise KeyError(f"Column '{column}' is not present in {family}")
            values.append(record[column])
        return compatible_array(values, family, column)


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


class AnalysisJsonlWriter:
    """Write one schema-described JSON array per frame for each result family."""

    def __init__(self, run, schemas, flush=True):
        self.run = run
        self.schemas = schemas
        self.flush = flush
        self.files = {}
        self.runtime_metadata = {"quantum": {}}

    def __enter__(self):
        paths = {
            "classical": self.run.classical_file,
            "classical_interactions": self.run.classical_interactions_file,
            "quantum": self.run.quantum_file,
            "quantum_interactions": self.run.quantum_interactions_file,
        }
        for family, path in paths.items():
            if not schema_has_data_columns(self.schemas[family]):
                continue
            if path is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files[family] = path.open("a")
        return self

    def __exit__(self, exc_type, exc, tb):
        for fileobj in self.files.values():
            fileobj.close()
        if exc_type is None:
            self.update_manifest_runtime_metadata()

    def write_frame(
        self,
        frame,
        classical=None,
        classical_interactions=None,
        quantum=None,
        quantum_interactions=None,
    ):
        results_by_family = {
            "classical": classical or [],
            "classical_interactions": classical_interactions or [],
            "quantum": quantum or [],
            "quantum_interactions": quantum_interactions or [],
        }
        if quantum:
            self.runtime_metadata["quantum"].update(runtime_quantum_metadata(quantum))

        for family, results in results_by_family.items():
            fileobj = self.files.get(family)
            if fileobj is None:
                continue
            schema = self.schemas[family].to_manifest()
            row = family_row(family, schema, frame, results, self.run.units)
            fileobj.write(json.dumps(row) + "\n")
            if self.flush:
                fileobj.flush()

    def update_manifest_runtime_metadata(self):
        if not self.runtime_metadata.get("quantum"):
            return
        with self.run.manifest_file.open() as f:
            manifest = json.load(f)
        manifest.setdefault("runtime_metadata", {}).update(self.runtime_metadata)
        with self.run.manifest_file.open("w") as f:
            json.dump(manifest, f, indent=2)


def prepare_output_files(config, config_file=None):
    run = create_analysis_run(config, config_file=config_file)
    reset_output_files(run)

    return run


def reset_output_files(run):
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


def analysis_run_in_directory(run, directory):
    directory = Path(directory)
    child = AnalysisRun(
        directory=directory,
        quantum_file=_child_output_path(run.directory, directory, run.quantum_file),
        quantum_interactions_file=_child_output_path(
            run.directory,
            directory,
            run.quantum_interactions_file,
        ),
        classical_interactions_file=_child_output_path(
            run.directory,
            directory,
            run.classical_interactions_file,
        ),
        classical_file=_child_output_path(run.directory, directory, run.classical_file),
        manifest_file=run.manifest_file,
        config_file=run.config_file,
        units=run.units,
    )
    return reset_output_files(child)


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


def write_manifest(config, run, trajectories=None):
    schemas = build_result_schemas(config)
    active_families = active_result_families(schemas)
    outputs = _manifest_outputs(run, active_families)
    has_child_directories = trajectories is not None and any(
        item["analysis_directory"] is not None
        for item in trajectories
    )
    if has_child_directories:
        outputs = {
            "per_trajectory": [
                _manifest_child_outputs(run, item, active_families)
                for item in trajectories
            ]
        }

    manifest = {
        "analysis_format_version": ANALYSIS_FORMAT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "directory": str(run.directory),
        "config_file": str(run.config_file),
        "trajectory": config.get("trajectory", {}),
        "trajectories": _manifest_trajectories(trajectories),
        "units": config.get("analysis", {}).get("units", DEFAULT_ANALYSIS_UNITS),
        "outputs": outputs,
        "schemas": schemas_to_manifest(schemas),
        "metadata": schema_metadata_to_manifest(schemas),
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


def _manifest_trajectories(trajectories):
    if trajectories is None:
        return []

    return [
        {
            "trajectory_index": item["trajectory_index"],
            "structure": item["structure"],
            "structure_directory": item["structure_directory"],
            "topology_file": str(item["topology_file"]),
            "trajectory_file": str(item["trajectory_file"]),
            "analysis_directory": item["analysis_directory"],
        }
        for item in trajectories
    ]


def _manifest_outputs(run, active_families):
    paths = {
        "quantum": run.quantum_file,
        "classical": run.classical_file,
        "quantum_interactions": run.quantum_interactions_file,
        "classical_interactions": run.classical_interactions_file,
    }
    return {family: str(path) for family, path in paths.items() if family in active_families}


def _manifest_child_outputs(run, item, active_families):
    directory = run.directory / item["analysis_directory"]
    entry = {
        "trajectory_index": item["trajectory_index"],
        "structure": item["structure"],
        "structure_directory": item["structure_directory"],
        "directory": str(directory),
    }
    child_paths = {
        "quantum": run.quantum_file,
        "classical": run.classical_file,
        "quantum_interactions": run.quantum_interactions_file,
        "classical_interactions": run.classical_interactions_file,
    }
    for family, path in child_paths.items():
        if family in active_families:
            entry[family] = str(_child_output_path(run.directory, directory, path))
    return entry


def _child_output_path(parent_directory, child_directory, path):
    try:
        relative = path.relative_to(parent_directory)
    except ValueError:
        relative = Path(path.name)
    return child_directory / relative


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


def append_quantum_results(target, results, metadata=None):
    if _append_schema_family(target, "quantum", results):
        return
    path = quantum_output_file(target)
    if path is None or not results:
        return

    with path.open("a") as f:
        for result in results:
            record = quantum_result_record(result, units=_analysis_units(target))
            f.write(json.dumps(_with_metadata(record, metadata)) + "\n")


def append_interaction_results(target, results, metadata=None):
    append_quantum_interaction_results(
        target,
        [result for result in results if result.type == "coupling"],
        metadata=metadata,
    )
    append_classical_interaction_results(
        target,
        [result for result in results if result.type in {"axis_angle", "distance", "orientation_factor", "plane_angle"}],
        metadata=metadata,
    )


def append_quantum_interaction_results(target, results, metadata=None):
    if _append_schema_family(target, "quantum_interactions", results):
        return
    path = quantum_interactions_output_file(target)
    if path is None or not results:
        return

    with path.open("a") as f:
        for result in results:
            record = quantum_interaction_result_record(result, units=_analysis_units(target))
            f.write(json.dumps(_with_metadata(record, metadata)) + "\n")


def append_classical_interaction_results(target, results, metadata=None):
    if _append_schema_family(target, "classical_interactions", results):
        return
    path = classical_interactions_output_file(target)
    if path is None or not results:
        return

    with path.open("a") as f:
        for result in results:
            record = classical_interaction_result_record(result, units=_analysis_units(target))
            f.write(json.dumps(_with_metadata(record, metadata)) + "\n")


def append_classical_results(target, results, metadata=None):
    if _append_schema_family(target, "classical", results):
        return
    path = classical_output_file(target)
    if path is None or not results:
        return

    with path.open("a") as f:
        for result in results:
            record = classical_result_record(result, units=_analysis_units(target))
            f.write(json.dumps(_with_metadata(record, metadata)) + "\n")


def _append_schema_family(target, family, results):
    if not isinstance(target, AnalysisRun) or not results:
        return False
    with target.manifest_file.open() as f:
        manifest = json.load(f)
    if not _is_schema_format(manifest):
        return False

    schema = manifest["schemas"][family]
    if not schema_has_data_columns(schema):
        return True
    frame = results[0].frame
    row = family_row(family, schema, frame, results, target.units)
    path = {
        "classical": target.classical_file,
        "classical_interactions": target.classical_interactions_file,
        "quantum": target.quantum_file,
        "quantum_interactions": target.quantum_interactions_file,
    }[family]
    if path is None:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return True


def _with_metadata(record, metadata):
    if not metadata:
        return record
    return {**metadata, **record}


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

    if _is_schema_format(manifest):
        return _load_schema_analysis_run(directory, manifest)

    return _load_record_analysis_run(directory, manifest)


def _is_schema_format(manifest):
    return (
        manifest.get("analysis_format_version") == ANALYSIS_FORMAT_VERSION
        and isinstance(manifest.get("schemas"), dict)
    )


def _load_schema_analysis_run(directory, manifest):
    outputs = manifest.get("outputs", {})
    if "per_trajectory" in outputs:
        per_trajectory = outputs["per_trajectory"]
        return LoadedAnalysisRun(
            directory=directory,
            manifest=manifest,
            quantum=_read_per_trajectory_schema_outputs(directory, manifest, per_trajectory, "quantum"),
            classical=_read_per_trajectory_schema_outputs(directory, manifest, per_trajectory, "classical"),
            quantum_interactions=_read_per_trajectory_schema_outputs(
                directory,
                manifest,
                per_trajectory,
                "quantum_interactions",
            ),
            classical_interactions=_read_per_trajectory_schema_outputs(
                directory,
                manifest,
                per_trajectory,
                "classical_interactions",
            ),
        )

    return LoadedAnalysisRun(
        directory=directory,
        manifest=manifest,
        quantum=_read_schema_outputs(directory, manifest, outputs, "quantum"),
        classical=_read_schema_outputs(directory, manifest, outputs, "classical"),
        quantum_interactions=_read_schema_outputs(directory, manifest, outputs, "quantum_interactions"),
        classical_interactions=_read_schema_outputs(directory, manifest, outputs, "classical_interactions"),
    )


def _load_record_analysis_run(directory, manifest):
    outputs = manifest.get("outputs", {})
    if "per_trajectory" in outputs:
        per_trajectory = outputs["per_trajectory"]
        return LoadedAnalysisRun(
            directory=directory,
            manifest=manifest,
            quantum=_read_per_trajectory_outputs(directory, per_trajectory, "quantum"),
            classical=_read_per_trajectory_outputs(directory, per_trajectory, "classical"),
            quantum_interactions=_read_per_trajectory_outputs(
                directory,
                per_trajectory,
                "quantum_interactions",
            ),
            classical_interactions=_read_per_trajectory_outputs(
                directory,
                per_trajectory,
                "classical_interactions",
            ),
        )

    return LoadedAnalysisRun(
        directory=directory,
        manifest=manifest,
        quantum=read_jsonl(_analysis_output_path(directory, outputs, "quantum")),
        classical=read_jsonl(_analysis_output_path(directory, outputs, "classical")),
        quantum_interactions=read_jsonl(_analysis_output_path(directory, outputs, "quantum_interactions")),
        classical_interactions=read_jsonl(_analysis_output_path(directory, outputs, "classical_interactions")),
    )


def _read_per_trajectory_outputs(directory, outputs, key):
    records = []
    for item in outputs:
        path = _analysis_output_path(directory, item, key)
        records.extend(read_jsonl(path))
    return records


def _read_per_trajectory_schema_outputs(directory, manifest, outputs, key):
    records = []
    for item in outputs:
        path = _analysis_output_path(directory, item, key)
        metadata = _loaded_trajectory_metadata(item)
        records.extend(_read_schema_outputs(directory, manifest, item, key, metadata=metadata))
    return records


def _read_schema_outputs(directory, manifest, outputs, key, metadata=None):
    path = _analysis_output_path(directory, outputs, key)
    schema = manifest.get("schemas", {}).get(key, {})
    rows = read_jsonl(path)
    records = []
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            raise ValueError(f"Expected JSON array row {line_number} in {path}")
        validate_row_length(row, schema, key)
        record = {
            column["name"]: value
            for column, value in zip(schema.get("columns", []), row)
        }
        if metadata:
            record = {**metadata, **record}
        records.append(record)
    return records


def _loaded_trajectory_metadata(item):
    return {
        "trajectory_index": item.get("trajectory_index"),
        "structure": item.get("structure"),
        "structure_directory": item.get("structure_directory"),
    }


def _loaded_dataframe(run, family, flatten=False):
    import pandas as pd

    records = _family_records(run, family)
    if _is_schema_format(run.manifest) and not flatten:
        return pd.DataFrame(records)
    return records_dataframe(records)


def _family_records(run, family):
    if family not in RESULT_FAMILIES:
        raise ValueError(f"Unsupported result family '{family}'")
    return getattr(run, family)


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
        ("center_of_geometry", "center_of_mass", "radius_of_gyration", "plane_rmsd"),
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
