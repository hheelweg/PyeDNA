"""Render HADDOCK configuration files for structure docking."""

from pathlib import Path
import re


DEFAULT_DOCKING_CONFIG = {
    "run_dir": "haddock/run",
    "mode": "local",
    "ncores": 32,
    "clean": False,
    "postprocess": False,

    "delenph": True,
    "autohis": False,

    "rigidbody_sampling": 10,
    "rigidbody_ntrials": 10,
    "rigidbody_randremoval": False,
    "rigidbody_unambig_scale": 800,
    "rigidbody_inter_rigid": 0.001,
    "rigidbody_elecflag": True,
    "rigidbody_w_air": 9999.0,
    "rigidbody_w_vdw": 1.0,
    "rigidbody_w_elec": 1.0,
    "rigidbody_w_desolv": 0.0,
    "rigidbody_w_bsa": 0.0,
    "rigidbody_w_dist": 9999.0,
    "rigidbody_cmrest": False,
    "rigidbody_surfrest": False,
    "rigidbody_ranair": False,
    "rigidbody_rigidtrans": True,

    "seletop_select": 10,

    "flexref_randremoval": False,
    "flexref_unambig_hot": 1000,
    "flexref_unambig_cool1": 1000,
    "flexref_unambig_cool2": 1000,
    "flexref_unambig_cool3": 1000,
    "flexref_w_air": 9999.0,
    "flexref_w_vdw": 1.0,
    "flexref_w_elec": 1.0,
    "flexref_w_desolv": 0.0,
    "flexref_w_bsa": 0.0,
    "flexref_mdsteps_rigid": 0,
    "flexref_mdsteps_cool1": 0,
    "flexref_mdsteps_cool2": 2000,
    "flexref_mdsteps_cool3": 2000,
    "flexref_dnarest_on": True,
    "flexref_tadfactor": 1,
    "flexref_temp_cool3_init": 300,
    "flexref_elecflag": True,

    "caprieval_allatoms": True,
}


def _flatten_docking_overrides(sections):
    """Validate and flatten sectioned HADDOCK overrides for template rendering."""

    prefixes = {
        "general": "",
        "topoaa": "",
        "rigidbody": "rigidbody_",
        "seletop": "seletop_",
        "flexref": "flexref_",
        "caprieval": "caprieval_",
    }

    values = {}

    for section, params in sections.items():
        if section not in prefixes:
            raise ValueError(
                f"Unknown HADDOCK override section [haddock.overrides.{section}]"
            )
        if not isinstance(params, dict):
            raise ValueError(
                f"[haddock.overrides.{section}] must contain key-value pairs"
            )

        prefix = prefixes[section]
        values.update({f"{prefix}{key}": value for key, value in params.items()})

    unknown = sorted(set(values) - set(DEFAULT_DOCKING_CONFIG))
    if unknown:
        raise KeyError(f"Unknown HADDOCK configuration parameters: {unknown}")

    return values


def _write_docking_config(dna_pdb, instances, top_file, par_file, restraint_file,
                          workdir=".", override_values=None, template=None):
    workdir = Path(workdir)
    output = workdir / "docking_config.cfg"

    if template is None:
        template = (
            Path(__file__).resolve().parents[2]
            / "templates"
            / "haddock_templates"
            / "docking_config.cfg"
        )
    else:
        template = Path(template)

    dna_pdb, top_file, par_file, restraint_file = map(Path, (dna_pdb, top_file, par_file, restraint_file))

    required = [template, dna_pdb, top_file, par_file, restraint_file]
    required += [instance.pdb for instance in instances]
    missing = [str(path) for path in required if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required HADDOCK files: {missing}")

    values = dict(DEFAULT_DOCKING_CONFIG)
    values.update(override_values or {})

    molecules = [dna_pdb] + [instance.pdb for instance in instances]

    values.update(
        topology_file=str(top_file),
        parameter_file=str(par_file),
        restraint_file=str(restraint_file),
        molecule_lines=",\n".join(f'    "{path}"' for path in molecules),
        flexibility_lines="\n\n".join(
            f'fle_seg_{i} = "{instance.segid}"\nfle_sta_{i} = 1\nfle_end_{i} = 1'
            for i, instance in enumerate(instances, start=1)
        ),
    )

    def format_value(value):
        return str(value).lower() if isinstance(value, bool) else str(value)

    text = template.read_text()
    for key, value in values.items():
        text = text.replace(f"{{{{ {key} }}}}", format_value(value))

    unresolved = sorted(set(re.findall(r"\{\{\s*(.*?)\s*\}\}", text)))
    if unresolved:
        raise KeyError(f"Missing docking template values: {unresolved}")

    output.write_text(text)
    print(f"Wrote {output}")

    return output
