import shutil
import subprocess
from pathlib import Path

from pyedna.structure import StructureBuilder


def run_structure(stage, config_file="structure.toml"):
    config_path = Path(config_file).resolve()

    builder = StructureBuilder.from_file(
        config_path,
        workdir=Path.cwd(),
    )

    if stage == "prepare":
        builder.prepare()

    elif stage == "dock":
        docking_config = Path("docking_config.cfg")

        if not docking_config.is_file():
            raise FileNotFoundError("docking_config.cfg not found")

        run_dir = Path("haddock/run")
        if run_dir.exists():
            shutil.rmtree(run_dir)

        subprocess.run(
            ["haddock3", str(docking_config)],
            check=True,
        )

    elif stage == "finalize":
        builder.finalize()

    elif stage == "amber":
        builder.prepare_amber()

    else:
        raise ValueError(f"Unknown structure stage: {stage}")