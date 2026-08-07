from pathlib import Path

from pyedna.haddock import select_best_models
from pyedna.structure_config import StructureConfig


def main():
    
    workdir = Path.cwd()
    config = StructureConfig.from_file(workdir / "struc.params")

    select_best_models(
        run_dir=workdir / "haddock" / "run",
        output_dir=workdir / "structures",
        top=config.top_models)


if __name__ == "__main__":
    main()