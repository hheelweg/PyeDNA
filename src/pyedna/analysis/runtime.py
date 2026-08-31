"""Runtime resource detection for trajectory analysis."""

from __future__ import annotations

from dataclasses import dataclass
import os


_CUDA_DISABLED_VALUES = {"", "-1", "none", "void", "nodevfiles"}


@dataclass(frozen=True)
class RuntimeResources:
    """Scheduler resources visible to the current analysis process."""

    num_cpus: int
    gpu_ids: list[str]

    @property
    def num_gpus(self):
        return len(self.gpu_ids)

    @property
    def has_gpu(self):
        return self.num_gpus > 0


def detect_runtime_resources(env=None):
    """Return CPU/GPU resources exposed by SLURM and the process environment."""

    env = os.environ if env is None else env
    return RuntimeResources(
        num_cpus=_slurm_cpus_per_task(env),
        gpu_ids=_visible_gpu_ids(env),
    )


def configure_thread_environment(num_threads):
    """Set BLAS/OpenMP thread counts for the current process."""

    value = str(max(1, int(num_threads)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = value


def gpu4pyscf_available():
    """Return whether GPU4PySCF and CuPy can use a visible CUDA device."""

    if not detect_runtime_resources().has_gpu:
        return False

    try:
        import cupy as cp
    except ImportError:
        return False

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
    except Exception:
        return False

    try:
        import gpu4pyscf  # noqa: F401
    except ImportError:
        return False

    return True


def require_gpu4pyscf():
    """Raise a clear error when GPU resources exist but GPU4PySCF is unusable."""

    if gpu4pyscf_available():
        return

    raise RuntimeError(
        "A GPU allocation is visible, but GPU4PySCF/CuPy is not available to "
        "the current process. Use a CPU allocation or load the validated CUDA, "
        "CuPy, and GPU4PySCF environment."
    )


def _slurm_cpus_per_task(env):
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        try:
            value = int(env.get(key, ""))
        except ValueError:
            continue
        if value > 0:
            return value
    return os.cpu_count() or 1


def _visible_gpu_ids(env):
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        return [
            token
            for token in (item.strip() for item in cuda_visible.split(","))
            if token.lower() not in _CUDA_DISABLED_VALUES
        ]

    if _running_under_slurm(env):
        count = _slurm_gpu_count(env)
        return [str(index) for index in range(count)]

    return []


def _running_under_slurm(env):
    return any(key.startswith("SLURM_") for key in env)


def _slurm_gpu_count(env):
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        count = _parse_gpu_count(env.get(key))
        if count:
            return count

    job_gpus = env.get("SLURM_JOB_GPUS")
    if not job_gpus:
        return 0
    return len([token for token in job_gpus.split(",") if token.strip()])


def _parse_gpu_count(value):
    if not value:
        return 0
    try:
        count = int(value)
    except ValueError:
        return 0
    return max(count, 0)
