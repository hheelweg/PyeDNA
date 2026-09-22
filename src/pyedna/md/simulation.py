"""Amber MD input generation and stage orchestration."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess

from pyedna.config import amber_environment, amber_executable

from .config import MDConfig
from .preparation import AmberSetup
from .restraints import AmberRestraintResolver
from .runtime import md_executable, slurm_ntasks, visible_gpus
from .structures import resolve_structure_runs, write_manifest, write_status


class MDSimulation:
    """Run one MD submission across the structures selected in md.toml."""

    def __init__(self, config, workdir=".", config_file=None, run_timestamp=None):
        self.config = config
        self.workdir = Path(workdir)
        self.config_file = Path(config_file) if config_file is not None else None
        timestamp = run_timestamp or datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.output_root = self.workdir / self.config.output.directory
        self.output_dir = self.output_root / f"run_{timestamp}"
        self.name = self.config.system.name
        self.structure_runs = resolve_structure_runs(self.config, self.workdir)

    @classmethod
    def from_file(cls, path, workdir="."):
        """Create an MD simulation from md.toml."""

        path = Path(path)
        return cls(MDConfig.from_file(path), workdir=workdir, config_file=path.resolve())

    def run(self):
        """Run the configured workflow for all selected structures."""

        self._resolve_md_engine()
        backend = self._backend_label()
        print(f"MD backend: {backend}", flush=True)
        print(f"Amber engine: {self.md_engine}", flush=True)

        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._copy_config()
        self._write_manifest()
        for run in self.structure_runs:
            structure_dir = self.output_dir / run["directory"]
            structure_dir.mkdir()
            write_status(structure_dir, run["structure"], "pending", "")

        gpus = visible_gpus()
        env = self._model_env(gpus[0]) if self.md_engine == "pmemd.cuda" and gpus else None
        for run in self.structure_runs:
            self._run_model(run, env=env)

        return self

    def _run_model(self, run, env=None):
        structure_dir = self.output_dir / run["directory"]
        simulation = _ModelSimulation(
            config=self.config,
            structure_index=run["structure"],
            structure_path=run["structure_path"],
            output_dir=structure_dir,
            source_workdir=self.workdir,
            md_engine=self.md_engine,
            md_engine_path=self.md_engine_path,
            env=env,
        )
        try:
            simulation.run()
        except Exception:
            write_status(
                structure_dir,
                run["structure"],
                "failed",
                simulation.current_stage,
            )
            raise

    def _copy_config(self):
        if self.config_file is not None:
            shutil.copy2(self.config_file, self.output_dir / self.config_file.name)

    def _write_manifest(self):
        write_manifest(
            self.output_dir / "manifest.toml",
            self.name,
            self.structure_runs,
            self.config.system,
        )

    @staticmethod
    def _model_env(gpu):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return env

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


class _ModelSimulation:
    """Run Amber preparation and MD stages for one selected structure."""

    def __init__(self, config, structure_index, structure_path, output_dir, source_workdir,
                 md_engine, md_engine_path, env=None):
        self.config = config
        self.structure_index = structure_index
        self.structure_path = Path(structure_path)
        self.output_dir = Path(output_dir)
        self.source_workdir = Path(source_workdir)
        self.name = self.config.system.name

        self.temp = self.config.simulation.temperature
        self.pressure = self.config.simulation.pressure
        self.dt = self.config.simulation.timestep
        self.traj_dt = self.config.traj_dt
        self.total_time = self.config.total_time
        self.traj_steps = self.config.traj_steps
        self.total_steps = self.config.production.steps

        self.prmtop = self.output_dir / f"{self.name}.prmtop"
        self.rst7 = self.output_dir / f"{self.name}.rst7"
        self.prmtop_name = self.prmtop.name
        self.rst7_name = self.rst7.name
        self.restraints = None
        self.md_engine = md_engine
        self.md_engine_path = md_engine_path
        self.env = env
        self.current_stage = ""

    def run(self):
        """Run the configured user-facing workflow stages."""

        if self.config.system.prmtop is not None:
            if "prepare" in self.config.workflow.stages:
                raise ValueError(
                    "'system.prmtop' supplies a ready topology; remove 'prepare' "
                    "from 'workflow.stages'"
                )
            self.copy_topology()

        for stage in self.config.workflow.stages:
            self._write_status("running", stage)
            self.current_stage = stage
            if stage == "prepare":
                self.run_preparation()
            if stage == "minimize":
                self.run_minimization()
            elif stage == "equilibrate":
                self.run_equilibration()
            elif stage == "production":
                self.run_production()

        self.clean_files()
        self._write_status("completed", "")
        return self

    def run_preparation(self):
        setup = AmberSetup.from_md_config(
            self.config,
            self.structure_index,
            workdir=self.output_dir,
            source_workdir=self.source_workdir,
        )
        setup.prepare(run_tleap=True, cleanup_intermediates=False)
        self._require_runtime_file(self.prmtop_name)
        self._require_runtime_file(self.rst7_name)
        self.restraints = AmberRestraintResolver(self.prmtop, self.config)
        print(self.restraints.analysis_text(), flush=True)

    def copy_topology(self):
        """Use a user-supplied prmtop/rst7 instead of the tleap 'prepare' stage."""

        for source, target in (
            (self.config.system.prmtop, self.prmtop),
            (self.config.system.rst7, self.rst7),
        ):
            source = Path(source)
            if not source.is_absolute():
                source = self.source_workdir / source
            if not source.exists():
                raise FileNotFoundError(f"Supplied topology file not found: {source}")
            shutil.copy2(source, target)
            print(f"Copied {source} -> {target}", flush=True)
        self.restraints = AmberRestraintResolver(self.prmtop, self.config)
        print(self.restraints.analysis_text(), flush=True)

    def _write_status(self, state, stage):
        write_status(self.output_dir, self.structure_index, state, stage)

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
        self._run_stage(
            stage="eq1",
            in_coord=f"min_{self.name}.ncrst",
            out_coord=f"eq1_{self.name}.ncrst",
            ref_coord=f"min_{self.name}.ncrst",
            netcdf=f"eq1_{self.name}.nc",
        )
        self._run_npt_equilibration()

    def _run_npt_equilibration(self):
        """Run NPT equilibration as one stage or restart chunks."""

        chunks = self.config.equilibration.npt_chunks
        if chunks == 1:
            self._write_input("eq2")
            self._run_stage(
                stage="eq2",
                in_coord=f"eq1_{self.name}.ncrst",
                out_coord=f"eq2_{self.name}.ncrst",
                ref_coord=f"min_{self.name}.ncrst",
                netcdf=f"eq2_{self.name}.nc",
            )
            return

        chunk_steps = self.config.equilibration.npt_steps // chunks
        in_coord = f"eq1_{self.name}.ncrst"
        final_restart = None
        previous_prefix = None
        (self.output_dir / f"eq2_{self.name}.out").write_text("")
        for index in range(1, chunks + 1):
            prefix = f"eq2_{index:03d}"
            self._write_input(
                "eq2",
                prefix=prefix,
                control_overrides={"nstlim": chunk_steps},
            )
            out_coord = f"{prefix}_{self.name}.ncrst"
            self._run_stage(
                stage="eq2",
                prefix=prefix,
                in_coord=in_coord,
                out_coord=out_coord,
                ref_coord=f"min_{self.name}.ncrst",
                netcdf=f"{prefix}_{self.name}.nc",
            )
            self._append_npt_chunk_output(prefix, index, chunks)
            in_coord = out_coord
            final_restart = out_coord
            if previous_prefix is not None:
                self._cleanup_npt_chunk(previous_prefix)
            previous_prefix = prefix

        shutil.copy2(
            self.output_dir / final_restart,
            self.output_dir / f"eq2_{self.name}.ncrst",
        )
        if self.config.equilibration.npt_cleanup in {"previous", "all"}:
            self._cleanup_npt_chunk(previous_prefix)

    def _append_npt_chunk_output(self, prefix, index, chunks):
        """Append one chunk log to the canonical eq2 output."""

        output = self.output_dir / f"eq2_{self.name}.out"
        chunk = self.output_dir / f"{prefix}_{self.name}.out"
        with output.open("a") as merged:
            merged.write(f"===== eq2 chunk {index}/{chunks}: {chunk.name} =====\n")
            merged.write(chunk.read_text())
            merged.write("\n")

    def _cleanup_npt_chunk(self, prefix):
        """Remove temporary files for a completed NPT chunk when configured."""

        if self.config.equilibration.npt_cleanup == "none":
            return
        for suffix in ("in", "out", "ncrst", "nc"):
            path = self.output_dir / f"{prefix}_{self.name}.{suffix}"
            if path.exists():
                path.unlink()

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

    def _write_input(self, stage, prefix=None, control_overrides=None):
        if self.restraints is None:
            self._require_runtime_file(self.prmtop_name)
            self.restraints = AmberRestraintResolver(self.prmtop, self.config)
        prefix = prefix or stage
        path = self.output_dir / f"{prefix}_{self.name}.in"
        path.write_text(self._stage_input(stage, control_overrides=control_overrides))
        return path

    def _stage_input(self, stage, control_overrides=None):
        restraint = self.restraints.for_stage(stage)
        titles = {
            "min1": "dna_dye: Initial minimization (solvent + ions)",
            "min2": "dna_dye: Initial minimization (entire system)",
            "eq1": "dna_dye: Heat system with restraint on DNA",
            "eq2": "dna_dye: NPT equilibration and slowly remove DNA restraint",
            "prod": "dna_dye: production run (NPT)",
        }
        controls = self._stage_controls(stage, restraint)
        if control_overrides:
            controls.update(control_overrides)
        lines = [titles[stage], self._namelist(controls)]
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
            "cut": self._stage_cutoff(stage),
            "ntr": int(restraint.active),
        }
        minimization = {
            "imin": 1,
            "maxcyc": min_cfg.max_steps,
            "ncyc": min_cfg.steepest_descent_steps,
            "ntmin": min_cfg.ntmin,
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

    def _stage_cutoff(self, stage):
        sim_cutoff = self.config.simulation.cutoff
        if stage in {"min1", "min2"}:
            return self.config.minimization.cutoff or sim_cutoff
        if stage == "eq1":
            return (
                self.config.equilibration.heating_cutoff
                or self.config.equilibration.cutoff
                or sim_cutoff
            )
        if stage == "eq2":
            return (
                self.config.equilibration.npt_cutoff
                or self.config.equilibration.cutoff
                or sim_cutoff
            )
        if stage == "prod":
            return self.config.production.cutoff or sim_cutoff
        raise ValueError(f"Unknown MD stage: {stage}")

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

    def _run_stage(self, stage, in_coord, out_coord, ref_coord, netcdf=None,
                   prefix=None):
        self._require_runtime_file(in_coord)
        self._require_runtime_file(ref_coord)
        if self.restraints is None:
            self.restraints = AmberRestraintResolver(self.prmtop, self.config)
        executable = self._stage_engine(stage)
        executable_path = self._stage_engine_path(executable)
        prefix = prefix or stage

        command = [
            *self._stage_launcher(executable, executable_path), "-O",
            "-i", f"{prefix}_{self.name}.in",
            "-o", f"{prefix}_{self.name}.out",
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
            env=self._amber_env(executable),
        )
        self._require_runtime_file(out_coord)
        if netcdf is not None:
            self._require_runtime_file(netcdf)

    def _md_engine(self):
        if self.md_engine is None:
            self._resolve_md_engine()
        return self.md_engine

    def _stage_engine(self, stage):
        if stage in {"min1", "min2"}:
            engine = self.config.minimization.engine
        elif stage in {"eq1", "eq2"}:
            engine = self.config.equilibration.engine
        elif stage == "prod":
            engine = self.config.production.engine
        else:
            raise ValueError(f"Unknown MD stage: {stage}")
        if engine == "auto":
            return self._md_engine()
        return engine

    def _stage_engine_path(self, executable):
        self._validate_stage_engine_resources(executable)
        if executable == self.md_engine:
            return self.md_engine_path
        return amber_executable(executable)

    def _validate_stage_engine_resources(self, executable):
        if executable == "pmemd.cuda" and not visible_gpus(self.env):
            print(
                "Warning: Amber engine 'pmemd.cuda' was requested but no CUDA "
                "GPU is visible to this process.",
                flush=True,
            )
            raise RuntimeError(
                "Amber engine 'pmemd.cuda' was requested for an MD stage, but no "
                "CUDA GPU is visible to the process. Request a GPU in the SLURM "
                "script, for example with '#SBATCH --gres=gpu:1', or set the "
                "stage engine to 'auto', 'pmemd', 'pmemd.MPI', or 'sander'."
            )
        if executable == "pmemd.MPI" and slurm_ntasks(self.env) <= 1:
            print(
                "Warning: Amber engine 'pmemd.MPI' was requested but "
                "SLURM_NTASKS is unset or 1; this stage will run with one MPI task.",
                flush=True,
            )

    def _amber_env(self, executable):
        env = amber_environment(executable)
        if self.env is not None and "CUDA_VISIBLE_DEVICES" in self.env:
            env["CUDA_VISIBLE_DEVICES"] = self.env["CUDA_VISIBLE_DEVICES"]
        return env

    def _stage_launcher(self, executable, executable_path):
        if executable == "pmemd.MPI":
            return [
                "mpirun",
                "-np",
                str(self._mpi_tasks(executable)),
                str(executable_path),
            ]
        return ["srun", "--ntasks", "1", str(executable_path)]

    def _mpi_tasks(self, executable):
        if executable == "pmemd.MPI":
            return slurm_ntasks(self.env)
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

        if cleanup == "standard":
            self._unlink_matching("eq2_[0-9][0-9][0-9]_*.in")
            self._unlink_matching("eq2_[0-9][0-9][0-9]_*.out")

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
