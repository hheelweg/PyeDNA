"""Runtime MD backend selection."""

from __future__ import annotations

import os


_CUDA_DISABLED_VALUES = {"", "-1", "none", "void", "nodevfiles"}


def gpu_available(env=None):
    """Return whether a CUDA GPU is visible to the current process."""

    env = os.environ if env is None else env
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return False

    tokens = [token.strip() for token in visible.split(",")]
    return any(token.lower() not in _CUDA_DISABLED_VALUES for token in tokens)


def md_executable(env=None):
    """Return the Amber engine selected from visible runtime resources."""

    env = os.environ if env is None else env
    if gpu_available(env):
        return "pmemd.cuda"
    if slurm_ntasks(env) > 1:
        return "pmemd.MPI"
    return "pmemd"


def slurm_ntasks(env=None):
    """Return the requested SLURM task count, defaulting to one task."""

    env = os.environ if env is None else env
    try:
        return int(env.get("SLURM_NTASKS", "1"))
    except ValueError:
        return 1
