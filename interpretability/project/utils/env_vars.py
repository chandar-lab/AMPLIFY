import importlib
import os
from logging import getLogger as get_logger
from pathlib import Path

import torch

logger = get_logger(__name__)


SLURM_JOB_ID: int | None = (
    int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
)
"""The value of the 'SLURM_JOB_ID' environment variable.

See https://slurm.schedmd.com/sbatch.html#OPT_SLURM_JOB_ID.
"""

SLURM_TMPDIR: Path | None = (
    Path(os.environ["SLURM_TMPDIR"])
    if "SLURM_TMPDIR" in os.environ
    else tmp
    if SLURM_JOB_ID is not None and (tmp := Path("/tmp")).exists()
    else None
)
"""The SLURM temporary directory, the fastest storage available.

- Extract your dataset to this directory at the start of your job.
- Remember to move any files created here to $SCRATCH since everything gets deleted at the end of the job.

See https://docs.mila.quebec/Information.html#slurm-tmpdir for more information.
"""

SCRATCH = Path(os.environ["SCRATCH"]) if "SCRATCH" in os.environ else None
"""Network directory where temporary logs / checkpoints / custom datasets should be saved.

Note that this is temporary storage. Files that you wish to be saved long-term should be saved to the `ARCHIVE` directory.

See https://docs.mila.quebec/Information.html#scratch for more information.
"""

ARCHIVE = Path(os.environ["ARCHIVE"]) if "ARCHIVE" in os.environ else None
"""Network directory for long-term storage. Only accessible from the login or cpu-only compute
nodes.

See https://docs.mila.quebec/Information.html#archive for more information.
"""


NETWORK_DATASETS_DIR = (
    Path(os.environ["NETWORK_DIR"])
    if "NETWORK_DIR" in os.environ
    else _mila_network_datasets_dir
    if (_mila_network_datasets_dir := Path("/network/datasets")).exists()
    else _drac_network_datasets_dir
    if (
        _drac_network_datasets_dir := Path.home() / "projects/rrg-bengioy-ad/data/curated"
    ).exists()
    else None
)
"""The (read-only) network directory that contains pre-downloaded datasets.

When running outside of the mila/DRAC clusters, this will be `None`, but can be mocked by setting the `NETWORK_DIR` environment variable.
"""

REPO_ROOTDIR = Path(__file__).parent
"""The root directory of this repository on this machine."""

for level in range(5):
    if "README.md" in list(p.name for p in REPO_ROOTDIR.iterdir()):
        break
    REPO_ROOTDIR = REPO_ROOTDIR.parent


DATA_DIR = Path(os.environ.get("DATA_DIR", (SLURM_TMPDIR or SCRATCH or REPO_ROOTDIR) / "data"))
"""Local Directory where datasets should be extracted on this machine."""


torchvision_dir: Path | None = None
"""Network directory with torchvision datasets."""
if (
    NETWORK_DATASETS_DIR
    and (_torchvision_dir := NETWORK_DATASETS_DIR / "torchvision").exists()
    and _torchvision_dir.is_dir()
):
    torchvision_dir = _torchvision_dir


def get_constant(*names: str):
    """Resolver for Hydra to get the value of a constant in this file."""
    assert names
    for name in names:
        if name in globals():
            obj = globals()[name]
            if obj is None:
                logger.debug(f"Value of {name} is None, moving on to the next value.")
                continue
            return obj
        parts = name.split(".")
        obj = importlib.import_module(parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        if obj is not None:
            return obj
        logger.debug(f"Value of {name} is None, moving on to the next value.")

    if len(names) == 1:
        raise RuntimeError(f"Could not find non-None value for name {names[0]}")
    raise RuntimeError(f"Could not find non-None value for names {names}")


NUM_WORKERS = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        os.environ.get(
            "SLURM_CPUS_ON_NODE",
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else torch.multiprocessing.cpu_count(),
        ),
    )
)
"""Default number of workers to be used by dataloaders, based on the number of CPUs and/or
tasks."""
