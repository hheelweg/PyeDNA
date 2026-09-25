"""Schema-based JSONL serialization for trajectory analysis results."""

from dataclasses import dataclass

from .units import distance_factor, energy_factor, merged_units


ANALYSIS_FORMAT_VERSION = 2
RESULT_FAMILIES = (
    "classical",
    "classical_interactions",
    "quantum",
    "quantum_interactions",
)


@dataclass(frozen=True)
class ResultSchema:
    columns: list
    metadata: list

    def to_manifest(self):
        return {"columns": self.columns}


def build_result_schemas(config):
    units = merged_units(config.get("analysis", {}).get("units"))
    return {
        "classical": build_classical_schema(config, units),
        "classical_interactions": build_classical_interactions_schema(config, units),
        "quantum": build_quantum_schema(config, units),
        "quantum_interactions": build_quantum_interactions_schema(config, units),
    }


def schemas_to_manifest(schemas):
    return {
        family: schema.to_manifest()
        for family, schema in schemas.items()
    }


def schema_has_data_columns(schema):
    if isinstance(schema, dict):
        columns = schema.get("columns", [])
    else:
        columns = getattr(schema, "columns", [])
    return any(column.get("name") != "frame" for column in columns)


def active_result_families(schemas):
    return [
        family for family, schema in schemas.items()
        if schema_has_data_columns(schema)
    ]


def schema_metadata_to_manifest(schemas):
    return {
        family: schema.metadata
        for family, schema in schemas.items()
    }


def build_classical_schema(config, units):
    columns = [_column("frame", "scalar")]
    metadata = []

    for job in config.get("classical", []):
        group = job["group"]
        outputs = job.get("outputs", [])
        result_outputs = []
        for output in outputs:
            for name, kind, shape, output_units in _classical_output_columns(output, units):
                columns.append(
                    _column(
                        f"{group}.{name}",
                        kind,
                        shape=shape,
                        units=output_units,
                    )
                )
                result_outputs.append(name)
        metadata.append({
            "id": group,
            "group": group,
            "outputs": result_outputs,
        })

    return ResultSchema(columns=columns, metadata=metadata)


def build_classical_interactions_schema(config, units):
    columns = [_column("frame", "scalar")]
    metadata = []

    for interaction in config.get("classical_interactions", []):
        groups = interaction["groups"]
        interaction_type = interaction["type"]
        method = interaction.get("method", _default_classical_interaction_method(interaction_type))
        outputs = _classical_interaction_outputs(interaction_type)
        prefix = _interaction_prefix(groups)

        for name in outputs:
            columns.append(
                _column(
                    f"{prefix}.{name}",
                    "scalar",
                    units=_classical_interaction_units(name, units),
                )
            )

        metadata.append({
            "id": f"{prefix}.{interaction_type}",
            "type": interaction_type,
            "groups": list(groups),
            "method": method,
            "outputs": outputs,
        })

    return ResultSchema(columns=columns, metadata=metadata)


def build_quantum_schema(config, units):
    columns = [_column("frame", "scalar")]
    metadata = []

    for job in config.get("quantum", []):
        group = job["group"]
        outputs = job.get("_write_outputs", job.get("outputs", []))
        result_outputs = []
        for output in outputs:
            for name, kind, shape, output_units in _quantum_output_columns(output, units):
                columns.append(
                    _column(
                        f"{group}.{name}",
                        kind,
                        shape=shape,
                        units=output_units,
                    )
                )
                result_outputs.append(name)

        metadata.append({
            "id": group,
            "group": group,
            "method": job.get("method"),
            "backend": job.get("backend"),
            "basis": job.get("basis"),
            "xc": job.get("xc"),
            "nstates": job.get("nstates"),
            "state_ids": job.get("state_ids"),
            "charge": job.get("charge"),
            "spin": job.get("spin"),
            "write_outputs": outputs,
            "outputs": result_outputs,
        })

    return ResultSchema(columns=columns, metadata=metadata)


def build_quantum_interactions_schema(config, units):
    columns = [_column("frame", "scalar")]
    metadata = []

    for interaction in config.get("quantum_interactions", []):
        groups = interaction["groups"]
        prefix = _interaction_prefix(groups)
        interaction_type = interaction["type"]
        method = interaction.get("method", "tdm")
        coupling_type = interaction.get("coupling_type", "electronic")
        state_pairs = interaction.get("state_pairs", [[0, 0]])

        for state_pair in state_pairs:
            columns.append(
                _column(
                    f"{prefix}.{_state_pair_name(state_pair)}.coupling",
                    "scalar",
                    units=units["coupling"],
                )
            )

        metadata.append({
            "id": f"{prefix}.{interaction_type}",
            "type": interaction_type,
            "groups": list(groups),
            "method": method,
            "state_pairs": [list(pair) for pair in state_pairs],
            "coupling_type": coupling_type,
            "outputs": ["coupling"],
        })

    return ResultSchema(columns=columns, metadata=metadata)


def family_row(family, schema, frame, results, units):
    values = {column["name"]: None for column in schema["columns"]}
    if "frame" in values:
        values["frame"] = frame

    units = merged_units(units)
    if family == "classical":
        for result in results:
            converted = convert_classical_values(result.values, units)
            for name, value in converted.items():
                values[f"{result.group}.{name}"] = to_json_value(value)
    elif family == "classical_interactions":
        for result in results:
            converted = convert_classical_interaction_values(result, units)
            prefix = _interaction_prefix(result.groups)
            for name, value in converted.items():
                values[f"{prefix}.{name}"] = to_json_value(value)
    elif family == "quantum":
        for result in results:
            converted = filtered_tddft_outputs(result, units=units)
            for name, value in converted.items():
                values[f"{result.group}.{name}"] = to_json_value(value)
    elif family == "quantum_interactions":
        for result in results:
            converted = convert_quantum_interaction_values(result, units)
            prefix = _interaction_prefix(result.groups)
            state = _state_pair_name(result.state_pair or [0, 0])
            for name, value in converted.items():
                values[f"{prefix}.{state}.{name}"] = to_json_value(value)
    else:
        raise ValueError(f"Unsupported result family: {family}")

    row = [values[column["name"]] for column in schema["columns"]]
    validate_row_length(row, schema, family)
    return row


def validate_row_length(row, schema, family):
    expected = len(schema["columns"])
    if len(row) != expected:
        raise ValueError(
            f"{family} row has {len(row)} values but schema defines {expected} columns"
        )


def row_to_record(row, schema):
    validate_row_length(row, schema, "analysis")
    return {
        column["name"]: value
        for column, value in zip(schema["columns"], row)
    }


def compatible_array(values, family, column):
    import numpy as np

    filtered = [value for value in values if value is not None]
    if not filtered:
        return np.asarray([])

    arrays = [np.asarray(value) for value in filtered]
    shape = arrays[0].shape
    mismatches = [array.shape for array in arrays if array.shape != shape]
    if mismatches:
        raise ValueError(
            f"Column '{column}' in {family} has incompatible array shapes; "
            f"expected {shape}, found {mismatches[0]}"
        )
    return np.stack(arrays)


def infer_kind(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return "scalar"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "object"


def runtime_quantum_metadata(results):
    metadata = {}
    for result in results:
        metadata[result.group] = {
            "atom_count": result.molecule.natm,
            "charge": result.molecule.charge,
            "spin": result.molecule.spin,
        }
    return metadata


def _column(name, kind, shape=None, units=None):
    column = {"name": name, "kind": kind}
    if units is not None:
        column["units"] = units
    if shape is not None:
        column["shape"] = shape
    return column


def _classical_output_columns(output, units):
    if output in {"center_of_geometry", "center_of_mass"}:
        return [(output, "array", [3], units["distance"])]
    if output == "radius_of_gyration":
        return [(output, "scalar", None, units["distance"])]
    if output == "plane_deviation":
        return [("plane_rmsd", "scalar", None, units["distance"])]
    return [(output, "object", None, None)]


def _classical_interaction_outputs(interaction_type):
    if interaction_type == "distance":
        return ["distance"]
    if interaction_type == "axis_angle":
        return ["axis_angle"]
    if interaction_type == "plane_angle":
        return ["plane_angle"]
    if interaction_type == "orientation_factor":
        return ["kappa", "kappa_squared"]
    return [interaction_type]


def _classical_interaction_units(name, units):
    if name == "distance":
        return units["distance"]
    if name in {"axis_angle", "plane_angle"}:
        return "degree"
    return None


def _default_classical_interaction_method(interaction_type):
    if interaction_type in {"axis_angle"}:
        return "axis"
    if interaction_type in {"plane_angle"}:
        return "plane"
    return "center_of_geometry"


def _quantum_output_columns(output, units):
    columns = []
    for key in output_keys_for_name(output):
        name = public_output_name(key, output)
        columns.append((name, *_quantum_output_descriptor(name, units)))
    return columns


def _quantum_output_descriptor(name, units):
    if name == "excited_state_energies":
        return "array", None, units["energy"]
    if name in {"oscillator_strengths"}:
        return "array", None, None
    if name in {"transition_dipoles"}:
        return "array", None, None
    if name in {"transition_quadrupoles", "tdm"}:
        return "array", None, None
    if name == "strongest_state":
        return "scalar", None, None
    if name in {"mulliken_populations", "mulliken_charges"}:
        return "array", None, None
    if name == "mulliken_fragments":
        return "object", None, None
    if name == "orbital_participation":
        return "object", None, None
    return "object", None, None


def filtered_tddft_outputs(result, units=None):
    requested = result.write_outputs or list(result.tddft)
    output = {}

    for name in requested:
        for key in output_keys_for_name(name):
            if key in result.tddft:
                public_name = public_output_name(key, name)
                value = convert_quantum_output(public_name, result.tddft[key], units)
                output[public_name] = to_json_value(value)

    return output


def convert_quantum_output(name, value, units):
    if name == "excited_state_energies":
        return scale_value(value, energy_factor(units["energy"]))
    return value


def convert_quantum_interaction_values(result, units):
    values = to_json_value(result.values)
    if result.type != "coupling":
        return values
    return scale_matching_keys(values, ("coupling",), energy_factor(units["coupling"]))


def convert_classical_interaction_values(result, units):
    values = to_json_value(result.values)
    if result.type != "distance":
        return values
    return scale_matching_keys(values, ("distance",), distance_factor(units["distance"]))


def convert_classical_values(values, units):
    values = to_json_value(values)
    return scale_matching_keys(
        values,
        ("center_of_geometry", "center_of_mass", "radius_of_gyration", "plane_rmsd"),
        distance_factor(units["distance"]),
    )


def scale_matching_keys(value, names, factor):
    if isinstance(value, dict):
        return {
            key: scale_value(item, factor) if matches_named_quantity(key, names) else scale_matching_keys(item, names, factor)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scale_matching_keys(item, names, factor) for item in value]
    return value


def matches_named_quantity(key, names):
    return any(key == name or key.startswith(f"{name} ") for name in names)


def scale_value(value, factor):
    value = to_json_value(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value * factor
    if isinstance(value, list):
        return [scale_value(item, factor) for item in value]
    if isinstance(value, dict):
        return {key: scale_value(item, factor) for key, item in value.items()}
    return value


def to_json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [to_json_value(item) for item in value]
    if isinstance(value, list):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "item"):
        return to_json_value(value.item())
    return value


def output_keys_for_name(name):
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


def public_output_name(key, requested_name):
    if key == "exc":
        return "excited_state_energies"
    if key == "osc":
        return "oscillator_strengths"
    if key == "idx":
        return "strongest_state"
    if key == "dip":
        return "transition_dipoles"
    if key == "quad":
        return "transition_quadrupoles"
    if key == "mull_pops":
        return "mulliken_populations"
    if key == "mull_chrgs":
        return "mulliken_charges"
    if key == "OPA":
        return "orbital_participation"
    return key


def _interaction_prefix(groups):
    return ".".join(groups)


def _state_pair_name(state_pair):
    return f"state_{state_pair[0]}_{state_pair[1]}"
