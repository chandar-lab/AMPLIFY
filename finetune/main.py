"""pipeline to start train/test"""

import copy
import logging
import os

import hydra
import numpy as np
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf

from downstreamer import downstream
from trainer import train

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.propagate = False


def prepare_task_config(cfg: DictConfig, task_name: str) -> DictConfig:
    """
    Prepares a configuration copy for each downstream task.

    Adjusts frozen_trunk, batch_size, and opt_interval settings based on the task name.

    Args:
        cfg: Global experiment configuration.
        task_name: The current downstream task name.

    Returns:
        A modified configuration copy specific for the current task.
    """

    # Constants
    SPECIAL_TASKS_FROZEN = [
        "Bo1015/ssp_q3",
        "Bo1015/fold_prediction",
        "Bo1015/contact_prediction_binary",
        "saprot_data/HumanPPI",
    ]
    CONTACT_PREDICTION_TASK = "Bo1015/contact_prediction_binary"

    cfg_copy = copy.deepcopy(cfg)
    # Allow dynamic modification of the configuration
    OmegaConf.set_struct(cfg_copy, False)

    # Freeze trunk for tasks in the special list
    cfg_copy.frozen_trunk = task_name in SPECIAL_TASKS_FROZEN

    if not cfg_copy.frozen_trunk:
        # For tasks with unfrozen trunk, halve the batch_size and double the opt_interval
        cfg_copy.batch_size = int(cfg_copy.batch_size / 2)
        cfg_copy.opt_interval = int(cfg_copy.opt_interval * 2)

    # Special handling for the contact prediction task
    if task_name == CONTACT_PREDICTION_TASK:
        cfg_copy.opt_interval = 64
        cfg_copy.batch_size = 2

    return cfg_copy


@hydra.main(version_base=None, config_path="config/", config_name="config.yaml")
def pipeline(cfg: DictConfig) -> None:
    """
    Args:
        cfg: configuration we use for this exp

    Returns: None

    """
    cfg_exp = cfg["experiments"]
    set_seed(cfg_exp.seed)

    if cfg_exp.mode == "train":
        logger.info("Starting training...")
        try:
            train(cfg_exp)
        except Exception as e:
            logger.exception("An error occurred during training: %s", e)

    elif cfg_exp.mode == "downstream":

        prt_model_safe = cfg_exp.prt_model_name.split("/")[-1]
        output_dir = os.path.join(
            "output",
            f"{prt_model_safe}",
            f"{cfg_exp.ft_model_path}",
            f"{cfg_exp.seed}"
        )

        task_names_todo_npy = os.path.join(output_dir, "task_names_todo.npy")
        # Load the list of pending tasks
        if os.path.exists(task_names_todo_npy):
            task_names_todo = list(np.load(task_names_todo_npy, allow_pickle=True))
        else:
            task_names_todo = cfg_exp.task_names

        for task_index, task_name in enumerate(task_names_todo):
            logger.info("Evaluate on downstream task %s start", task_name)
            # Prepare configuration for the current task
            cfg_task = prepare_task_config(cfg_exp, task_name)
            downstream(cfg_task, task_name)
            logger.info("Evaluate on downstream task %s end", task_name)
            np.save(task_names_todo_npy, task_names_todo[task_index + 1 :])

        logger.info("All downstream tasks finished")
    else:
        logger.error("Unknown mode: %s", cfg_exp.mode)


if __name__ == "__main__":
    pipeline()
