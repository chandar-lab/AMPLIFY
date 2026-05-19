"""Main entry point for interpretability experiments using Hydra configuration management.

This script orchestrates the training and evaluation pipeline:
1. Parses configuration files using Hydra
2. Instantiates components (trainer, algorithm, datamodule, network) from config
3. Trains the model (e.g., linear probe or SAE)
4. Runs evaluation loop and logs metrics

The configuration system uses Hydra's composition pattern, allowing flexible experiment
setup through YAML config files in `project/configs/`.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import hydra
import lightning
import rich
import rich.logging
import wandb
from hydra_plugins.auto_schema import auto_schema_plugin
from omegaconf import DictConfig

import project
from project.configs import add_configs_to_hydra_store
from project.configs.config import Config
from project.experiment import train_and_evaluate
from project.utils.hydra_utils import resolve_dictconfig
from project.utils.typing_utils import HydraConfigFor
from project.utils.utils import print_config

PROJECT_NAME = project.__name__
REPO_ROOTDIR = Path(__file__).parent.parent
logger = logging.getLogger(__name__)

# todo: remove, or configure in some other way (e.g. a config file).
auto_schema_plugin.config = auto_schema_plugin.AutoSchemaPluginConfig(
    schemas_dir=REPO_ROOTDIR / ".schemas",
    regen_schemas=False,
    stop_on_error=False,
    quiet=False,
    verbose=False,
    add_headers=False,  # don't fallback to adding headers if we can't use vscode settings file.
)

add_configs_to_hydra_store()


@hydra.main(
    config_path=f"pkg://{PROJECT_NAME}.configs",
    config_name="config",
    version_base="1.2",
)
def main(dict_config: DictConfig) -> dict:
    """Main entry point: trains & evaluates a learning algorithm."""

    print_config(dict_config, resolve=False)
    assert dict_config["algorithm"] is not None

    # Resolve all the interpolations in the configs.
    config: Config = resolve_dictconfig(dict_config)
    setup_logging(
        log_level=config.log_level,
        global_log_level="DEBUG" if config.debug else "INFO" if config.verbose else "WARNING",
    )

    # Seed the random number generators, so the weights that are
    # constructed are deterministic and reproducible.
    lightning.seed_everything(seed=config.seed, workers=True)

    if config.datamodule is None:
        datamodule = None
    elif isinstance(config.datamodule, lightning.LightningDataModule):
        # The datamodule was already instantiated for the `instance_attr` resolver to
        # get an attribute like `num_classes` when instantiating a network config.
        datamodule = config.datamodule
    else:
        datamodule = hydra.utils.instantiate(config.datamodule)

    # Create the algo.
    algorithm = instantiate_algorithm(config.algorithm, datamodule=datamodule)

    # Do the training and evaluation, returns the metric name and the overall 'error' to minimize.
    metric_name, error = train_and_evaluate(algorithm, config=config, datamodule=datamodule)

    if wandb.run:
        wandb.finish()

    assert error is not None
    # Results are returned like this so that the Orion sweeper can parse the results correctly.
    return dict(name=metric_name, type="objective", value=error)


def setup_logging(log_level: str, global_log_level: str = "WARNING") -> None:
    from project.main import PROJECT_NAME

    logging.basicConfig(
        level=global_log_level.upper(),
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[
            rich.logging.RichHandler(
                markup=True,
                rich_tracebacks=True,
                tracebacks_width=100,
                tracebacks_show_locals=False,
            )
        ],
    )

    project_logger = logging.getLogger(PROJECT_NAME)
    project_logger.setLevel(log_level.upper())


def instantiate_algorithm(
    algorithm_config: HydraConfigFor[lightning.LightningModule],
    datamodule: lightning.LightningDataModule | None = None,
) -> lightning.LightningModule:
    """Function used to instantiate the algorithm.

    It is suggested that your algorithm (LightningModule) take in the `datamodule` and `network`
    as arguments, to make it easier to swap out different networks and datamodules during
    experiments.

    The instantiated datamodule will be passed to the algorithm's constructor.
    """
    # Create the algorithm
    algo_config = algorithm_config

    if datamodule:
        algo_or_algo_partial = hydra.utils.instantiate(algo_config, datamodule=datamodule)
    else:
        algo_or_algo_partial = hydra.utils.instantiate(algo_config)

    if isinstance(algo_or_algo_partial, functools.partial):
        if datamodule:
            return algo_or_algo_partial(datamodule=datamodule)
        return algo_or_algo_partial()

    algorithm = algo_or_algo_partial
    return algorithm


if __name__ == "__main__":
    main()
