# https://github.com/facebookresearch/hydra/blob/main/examples/plugins/example_launcher_plugin/hydra_plugins/example_launcher_plugin/example_launcher.py

import dataclasses
import logging
import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import hydra_zen
from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides
from hydra.plugins.plugin import Plugin
from hydra.types import HydraContext, TaskFunction
from hydra.utils import instantiate
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher
from omegaconf import DictConfig
from remote_slurm_executor.slurm_remote import RemoteSlurmExecutor
from remote_slurm_executor.utils import LoginNode

logger = logging.getLogger(__name__)


def _instantiate(self: Plugins, config: DictConfig) -> Plugin:
    """FIX annoying Hydra thing: plugins have to be in an "approved" `hydra_plugins` namespace?!"""
    from hydra._internal import utils as internal_utils

    classname = internal_utils._get_cls_name(config, pop=False)
    try:
        if classname is None:
            raise ImportError("class not configured")

        if not self.is_in_toplevel_plugins_module(classname):
            # NOTE: This is the weirdly strict thing we're patching:
            # "All plugins must be defined inside the approved top level modules."
            # "For plugins outside of hydra-core, the approved module is hydra_plugins."
            # raise RuntimeError(f"Invalid plugin '{classname}': not the hydra_plugins package")
            plugin = instantiate(config)
            assert isinstance(plugin, Plugin)
            return plugin

        if classname not in self.class_name_to_class.keys():
            raise RuntimeError(f"Unknown plugin class : '{classname}'")
        clazz = self.class_name_to_class[classname]
        plugin = instantiate(config=config, _target_=clazz)
        assert isinstance(plugin, Plugin)

    except ImportError as e:
        raise ImportError(
            f"Could not instantiate plugin {classname} : {str(e)}\n\n\tIS THE PLUGIN INSTALLED?\n\n"
        )

    return plugin


Plugins._instantiate = _instantiate


# Made this a dataclass to avoid having an ugly default repr, but it causes issues with
# hydra-auto-schema because it tries to create a schema for everything here.
# @dataclasses.dataclass(init=False)
class RemoteSlurmLauncher(BaseSubmititLauncher):
    _EXECUTOR: ClassVar[str] = "remoteslurm"

    params: dict[str, Any]
    config: DictConfig | None = None
    task_function: TaskFunction | None = None
    sweep_configs: TaskFunction | None = None
    hydra_context: HydraContext | None = None
    executor: RemoteSlurmExecutor

    def __init__(
        self,
        executor: Callable[[], RemoteSlurmExecutor],
        account: str | None = None,
        array_parallelism: int = 256,
        comment: str | None = None,
        constraint: str | None = None,
        cpus_per_gpu: int | None = None,
        cpus_per_task: int | None = None,
        dependency: str | None = None,
        exclude: str | None = None,
        exclusive: bool | None = None,
        gpus_per_node: int | str | None = None,
        gpus_per_task: int | str | None = None,
        gres: str | None = None,
        # job_name: str = "submitit",
        job_name: str = "submitit-${hydra.job.name}",
        mail_type: str | None = None,
        mail_user: str | None = None,
        mem: str | None = None,
        mem_per_cpu: str | None = None,
        mem_per_gpu: str | None = None,
        nodelist: str | None = None,
        nodes: int = 1,
        ntasks_per_node: int | None = None,
        num_gpus: int | None = None,
        partition: str | None = None,
        qos: str | None = None,
        setup: list[str] | None = None,
        signal_delay_s: int = 90,
        srun_args: list[str] | None = None,
        stderr_to_stdout: bool = True,  # changed!
        time: str | int = 5,
        use_srun: bool = True,
        wckey: str = "submitit",
        additional_parameters: dict | None = None,
        tasks_per_node: int | None = None,
        mem_gb: int | None = None,
    ) -> None:
        setup = setup or []
        additional_parameters = additional_parameters or {}
        self.executor = executor()
        cluster = self.executor.cluster_hostname

        if account is None and len(available_accounts := get_slurm_accounts(cluster)) > 1:
            # NOTE: tends to favour rrg-*_gpu accounts on DRAC.
            account = sorted(available_accounts)[-1]
            warnings.warn(
                UserWarning(
                    f"The slurm account to use wasn't passed, and you have multiple accounts on "
                    f"the {cluster} cluster: {available_accounts}\n"
                    f"Will use --account={account} when launching jobs."
                )
            )

        if setup is not None and (executor_setup := self.executor.parameters.get("setup")):
            # The executor already has some lines in "setup", don't overwrite those later.
            setup = executor_setup + setup
        if mem_gb is not None:
            assert mem is None, "can't use both mem and mem_gb"
            mem = f"{mem_gb}GB"
        if tasks_per_node is not None:
            assert ntasks_per_node is None, "can't use both tasks_per_node and ntasks_per_node"
            ntasks_per_node = tasks_per_node
        if ntasks_per_node is not None:
            additional_parameters["ntasks-per-node"] = ntasks_per_node
        super().__init__(
            account=account,
            array_parallelism=array_parallelism,
            comment=comment,
            constraint=constraint,
            cpus_per_gpu=cpus_per_gpu,
            cpus_per_task=cpus_per_task,
            dependency=dependency,
            exclude=exclude,
            exclusive=exclusive,
            gpus_per_node=gpus_per_node,
            gpus_per_task=gpus_per_task,
            gres=gres,
            job_name=job_name,
            mail_type=mail_type,
            mail_user=mail_user,
            mem=mem,
            mem_per_cpu=mem_per_cpu,
            mem_per_gpu=mem_per_gpu,
            nodelist=nodelist,
            nodes=nodes,
            num_gpus=num_gpus,
            partition=partition,
            qos=qos,
            setup=setup,
            signal_delay_s=signal_delay_s,
            srun_args=srun_args,
            stderr_to_stdout=stderr_to_stdout,
            time=time,
            use_srun=use_srun,
            wckey=wckey,
            additional_parameters=additional_parameters,
        )
        self.executor.update_parameters(**self.params)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        # specify resources/parameters\
        # Do *not* overwrite the `setup` if it's already in the executor's parameters!

        # TODO: Make sure that if `launch` is called multiple times, `update_parameters` isn't (or
        # that it being called multiple times doesn't cause issues).

        logger.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : " f"{self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: list[tuple[list[str], str, int, str, dict]] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            logger.info(f"\t#{idx} : {lst}")
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )
        # NOTE: A bit weird that the executor is pickling itself here, but oh well.
        jobs = self.executor.map_array(self, *zip(*job_params))
        print(f"JOB IDS: {[j.job_id for j in jobs]}")
        # TODO: Here the results of all tasks other than task 0 is ignored. This is bad!
        # If we have `--ntasks-per-gpu`, then perhaps the other task results are different results
        # for different seeds, or something similar!
        return [j.results()[0] for j in jobs]

    def __call__(
        self,
        sweep_overrides: list[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: dict[type, Singleton],
    ) -> JobReturn:
        return super().__call__(
            sweep_overrides=sweep_overrides,
            job_dir_key=job_dir_key,
            job_num=job_num,
            job_id=job_id,
            singleton_state=singleton_state,
        )


# @functools.cache
def get_slurm_accounts(cluster: str) -> list[str]:
    """Gets the SLURM accounts of the user using sacctmgr on the slurm cluster."""
    logger.debug(f"Fetching the list of SLURM accounts available on the {cluster} cluster.")
    result = LoginNode(cluster).run(
        "sacctmgr --noheader show associations where user=$USER format=Account%50"
    )
    accounts = [line.strip() for line in result.stdout.splitlines()]
    assert accounts
    return accounts


RemoteSlurmQueueConf = hydra_zen.builds(
    RemoteSlurmLauncher,
    populate_full_signature=True,
    # zen_partial=True,
    hydra_convert="object",
    zen_dataclass={"cls_name": "RemoteSlurmQueueConf"},
)


from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf  # noqa
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher  # noqa
# from hydra_plugins.hydra_submitit_launcher. import SlurmLauncher  # noqa

from submitit.slurm.slurm import _make_sbatch_string  # noqa

# Interesting idea: Create the config based on the signature of that function directly.
_AddedArgumentsConf = hydra_zen.builds(
    _make_sbatch_string,
    populate_full_signature=True,
    hydra_convert="object",
    zen_exclude=["command", "folder", "map_count"],
)


@dataclasses.dataclass
class PatchedSlurmQueueConf(_AddedArgumentsConf, SlurmQueueConf):
    """Adds more SLURM parameters to the config for the SLURM submitit launcher of Hydra."""

    _target_: str = "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"  # type: ignore

    signal_delay_s: int = 120
    """USR1 signal delay before timeout."""

    max_num_timeout: int = 0
    """Maximum number of retries on job timeout.

    Change this only after you confirmed your code can handle re-submission by properly resuming
    from the latest stored checkpoint. check the following for more info on slurm_max_num_timeout
    https://github.com/facebookincubator/submitit/blob/master/docs/checkpointing.md
    """

    additional_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Useful to add parameters which are not currently available in the plugin.

    Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    """

    array_parallelism: int = 256
    """Maximum number of jobs running in parallel."""

    setup: list[str] | None = None
    """A list of commands to run in sbatch before running srun."""


ConfigStore.instance().store(
    group="hydra/launcher",
    name="patched_submitit_slurm",
    node=PatchedSlurmQueueConf,
    provider="Mila",
)
