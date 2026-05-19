from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger as get_logger

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

logger = get_logger(__name__)


# @rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "algorithm",
        "datamodule",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    TAKEN FROM https://github.com/ashleve/lightning-hydra-template/blob/6a92395ed6afd573fa44dd3a054a603acbdcac06/src/utils/__init__.py#L56

    Args:
        config: Configuration composed by Hydra.
        print_order: Determines in what order config components are printed.
        resolve: Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    for f in print_order:
        if f in config:
            queue.append(f)
        else:
            logger.info(f"Field '{f}' not found in config")

    for f in config:
        if f not in queue:
            queue.append(f)

    for f in queue:
        if f not in config:
            logger.info(f"Field '{f}' not found in config")
            continue
        branch = tree.add(f, style=style, guide_style=style)

        config_group = config[f]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    # with open("config_tree.log", "w") as file:
    #     rich.print(tree, file=file)
