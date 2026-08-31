"""Amber MD input generation and stage orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import subprocess

from pyedna.config import amber_environment, amber_executable

from .config import MDConfig
from .restraints import AmberRestraintResolver
from .runtime import md_executable, slurm_ntasks


class MDSimulation:
    """Run Amber minimization, equilibration, and production stages."""

    def __init__(self, config, workdir=".", config_file=None, run_timestamp=None):
        self.config = config
        self.workdir = Path(workdir)
        self.config_file = Path(config_file) if config_file is not None else None
        timestamp = run_timestamp or datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.output_root = self.workdir / self.config.output.directory
        self.output_dir = self.output_root / f"run_{timestamp}"
        self.name = self.config.system.name

        self.temp = self.config.simulation.temperature
        self.pressure = self.config.simulation.pressure
        self.dt = self.config.simulation.timestep
        self.traj_dt = self.config.traj_dt
        self.total_time = self.config.total_time
        self.traj_steps = self.config.traj_steps
        self.total_steps = self.config.production.steps

        self.prmtop = self._resolve_input(self.config.system.prmtop_path)
        self.rst7 = self._resolve_input(self.config.system.rst7_path)
        self.restraints = AmberRestraintResolver(self.prmtop, self.config)
        self.md_engine = None
        self.md_engine_path = None

    @classmethod
    def from_file(cls, path, workdir="."):
        """Create an MD simulation from md.toml."""

        path = Path(path)
        return cls(MDConfig.from_file(path), workdir=workdir, config_file=path.resolve())

    def _resolve_input(self, filename):
        path = Path(filename)
        if not path.is_absolute():
            path = self.workdir / path
        if not path.exists():
            raise FileNotFoundError(f"Amber input file not found: {path}")
        return path

    def run(self):
        """Run the configured user-facing workflow stages."""

        self._resolve_md_engine()
        backend = self._backend_label()
        print(f"MD backend: {backend}", flush=True)
        print(f"Amber engine: {self.md_engine}", flush=True)

        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._copy_config()
        self._link_or_copy_inputs()
        print(self.restraints.analysis_text(), flush=True)

        for stage in self.config.workflow.stages:
            if stage == "minimize":
                self.run_minimization()
            elif stage == "equilibrate":
                self.run_equilibration()
            elif stage == "production":
                self.run_production()

        self.clean_files()
        return self

    def _copy_config(self):
        if self.config_file is not None:
            shutil.copy2(self.config_file, self.output_dir / self.config_file.name)

    def _link_or_copy_inputs(self):
        for source in (self.prmtop, self.rst7):
            target = self.output_dir / source.name
            if target.exists():
                continue
            try:
                target.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, target)

        self.prmtop_name = self.prmtop.name
        self.rst7_name = self.rst7.name

    def run_minimization(self):
        """Run solvent/ion and whole-system minimization."""

        self._write_input("min1")
        self._write_input("min2")
        self._run_stage(
            stage="min1",
            in_coord=self.rst7_name,
            out_coord=f"min1_{self.name}.ncrst",
            ref_coord=self.rst7_name,
        )
        self._run_stage(
            stage="min2",
            in_coord=f"min1_{self.name}.ncrst",
            out_coord=f"min_{self.name}.ncrst",
            ref_coord=f"min1_{self.name}.ncrst",
        )

    def run_equilibration(self):
        """Run heating and NPT equilibration."""

        self._require_runtime_file(f"min_{self.name}.ncrst")
        self._write_input("eq1")
        self._write_input("eq2")
        self._run_stage(
            stage="eq1",
            in_coord=f"min_{self.name}.ncrst",
            out_coord=f"eq1_{self.name}.ncrst",
            ref_coord=f"min_{self.name}.ncrst",
            netcdf=f"eq1_{self.name}.nc",
        )
        self._run_stage(
            stage="eq2",
            in_coord=f"eq1_{self.name}.ncrst",
            out_coord=f"eq2_{self.name}.ncrst",
            ref_coord=f"min_{self.name}.ncrst",
            netcdf=f"eq2_{self.name}.nc",
        )

    def run_production(self):
        """Run production MD."""

        self._require_runtime_file(f"eq2_{self.name}.ncrst")
        self._write_input("prod")
        self._run_stage(
            stage="prod",
            in_coord=f"eq2_{self.name}.ncrst",
            out_coord=f"{self.name}.ncrst",
            ref_coord=f"min_{self.name}.ncrst",
            netcdf=f"{self.name}.nc",
        )

    def _write_input(self, stage):
        path = self.output_dir / f"{stage}_{self.name}.in"
        path.write_text(self._stage_input(stage))
        return path

    def _stage_input(self, stage):
        restraint = self.restraints.for_stage(stage)
        titles = {
            "min1": "dna_dye: Initial minimization (solvent + ions)",
            "min2": "dna_dye: Initial minimization (entire system)",
            "eq1": "dna_dye: Heat system with restraint on DNA",
            "eq2": "dna_dye: NPT equilibration and slowly remove DNA restraint",
            "prod": "dna_dye: production run (NPT)",
        }
        lines = [titles[stage], self._namelist(self._stage_controls(stage, restraint))]
        return "\n".join(lines) + "\n"

    def _stage_controls(self, stage, restraint):
        sim = self.config.simulation
        min_cfg = self.config.minimization
        eq = self.config.equilibration
        prod = self.config.production
        thermo = self.config.thermostat
        baro = self.config.barostat

        common = {
            "iwrap": sim.iwrap,
            "cut": sim.cutoff,
            "ntr": int(restraint.active),
        }
        minimization = {
            "imin": 1,
            "maxcyc": min_cfg.max_steps,
            "ncyc": min_cfg.steepest_descent_steps,
            "ntb": sim.ntb,
        }
        md = {
            "imin": 0,
            "dt": sim.timestep,
            "ntc": sim.ntc,
            "ntf": sim.ntf,
            "temp0": sim.temperature,
            "ntt": thermo.amber_ntt,
            "gamma_ln": thermo.gamma,
            "ig": thermo.seed,
            "ioutfm": sim.ioutfm,
        }
        npt = {
            "ntp": sim.ntp,
            "pres0": sim.pressure,
            "taup": baro.tau,
        }
        restraint_controls = self._mask_restraint_controls(restraint)

        if stage == "min1":
            return {**minimization, **common, **restraint_controls}
        if stage == "min2":
            return {**minimization, **common, **restraint_controls}
        if stage == "eq1":
            return {
                **md,
                **common,
                **restraint_controls,
                "nstlim": eq.heating_steps,
                "irest": 0,
                "ntx": 1,
                "ntb": sim.ntb,
                "tempi": sim.initial_temperature,
                "ntpr": eq.ntpr,
                "ntwx": eq.ntwx,
                "ntwr": eq.ntwr,
            }
        if stage == "eq2":
            return {
                **md,
                **npt,
                **common,
                **restraint_controls,
                "nstlim": eq.npt_steps,
                "irest": 1,
                "ntx": 5,
                "tempi": sim.temperature,
                "ntpr": eq.ntpr,
                "ntwx": eq.ntwx,
                "ntwr": eq.ntwr,
            }

        controls = {
            **md,
            **npt,
            **common,
            "nstlim": prod.steps,
            "irest": 1,
            "ntx": 5,
            "tempi": sim.temperature,
            "ntpr": prod.log_interval,
            "ntwx": prod.trajectory_interval,
            "ntwr": prod.restart_interval,
            "ntwf": prod.force_interval,
        }
        controls.update(restraint_controls)
        return controls

    def _mask_restraint_controls(self, restraint):
        if not restraint.active:
            return {}
        return {
            "restraint_wt": restraint.strength,
            "restraintmask": self._amber_quote(restraint.mask),
        }

    @staticmethod
    def _namelist(values):
        lines = [" &cntrl"]
        items = list(values.items())
        for index, (key, value) in enumerate(items):
            suffix = "," if index < len(items) - 1 else ""
            lines.append(f"  {key} = {value}{suffix}")
        lines.append("/")
        return "\n".join(lines)

    @staticmethod
    def _amber_quote(value):
        if value is None:
            return value
        value = str(value)
        if value.startswith("'") and value.endswith("'"):
            return value
        return f"'{value}'"

    def _run_stage(self, stage, in_coord, out_coord, ref_coord, netcdf=None):
        self._require_runtime_file(in_coord)
        self._require_runtime_file(ref_coord)
        executable = self._md_engine()

        command = [
            *self._stage_launcher(), "-O",
            "-i", f"{stage}_{self.name}.in",
            "-o", f"{stage}_{self.name}.out",
            "-p", self.prmtop_name,
            "-c", in_coord,
            "-r", out_coord,
            "-ref", ref_coord,
        ]
        if netcdf is not None:
            command.extend(["-x", netcdf])

        subprocess.run(
            command,
            cwd=self.output_dir,
            check=True,
            env=amber_environment(executable),
        )
        self._require_runtime_file(out_coord)
        if netcdf is not None:
            self._require_runtime_file(netcdf)

    def _md_engine(self):
        if self.md_engine is None:
            self._resolve_md_engine()
        return self.md_engine

    def _resolve_md_engine(self):
        self.md_engine = md_executable()
        try:
            self.md_engine_path = amber_executable(self.md_engine)
        except RuntimeError as exc:
            backend = self._backend_label()
            raise RuntimeError(
                f"Selected {backend} MD backend but {self.md_engine} is unavailable. "
                f"{exc}"
            ) from exc

    def _backend_label(self):
        if self.md_engine == "pmemd.cuda":
            return "GPU"
        if self.md_engine == "pmemd.MPI":
            return "CPU MPI"
        return "CPU"

    def _stage_launcher(self):
        if self._md_engine() == "pmemd.MPI":
            return ["mpirun", "-np", str(self._mpi_tasks()), str(self.md_engine_path)]
        return ["srun", "--ntasks", "1", str(self.md_engine_path)]

    def _mpi_tasks(self):
        if self._md_engine() == "pmemd.MPI":
            return slurm_ntasks()
        return 1

    def _require_runtime_file(self, filename):
        path = self.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required MD file not found: {path}")
        return path

    def clean_files(self):
        """Remove intermediate runtime files according to output.cleanup."""

        cleanup = self.config.output.cleanup
        if cleanup == "all":
            return

        if cleanup == "minimal":
            self._unlink_matching("*.in")

        if cleanup == "minimal":
            self._unlink_matching("*.out", keep={f"prod_{self.name}.out"})

        if cleanup in {"minimal", "standard"}:
            self._unlink_matching("*.ncrst", keep={f"min_{self.name}.ncrst"})
            self._unlink_matching("*.nc", keep={f"{self.name}.nc"})

        if cleanup == "restart":
            self._unlink_matching("*.nc", keep={f"{self.name}.nc"})

    def _unlink_matching(self, pattern, keep=None):
        keep = keep or set()
        for path in self.output_dir.glob(pattern):
            if path.name not in keep:
                path.unlink()
