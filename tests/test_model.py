import json
import os
import shutil

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import safetensors

from amplify.dataset import get_dataloader
from amplify.loss import get_loss
from amplify.tokenizer import ProteinTokenizer
from amplify.model.amplify import AMPLIFY
from amplify.trainer import trainer

from .fixtures import training_run_output_path

this_dir = os.path.abspath(os.path.dirname(__file__))

vocab_rel_path = "tests/example-data/easy-vocab.txt"
vocab_path = os.path.join(this_dir, "..", vocab_rel_path)


def test__esm__trains_on_an_easy_task_with_decreasing_train_set_loss(
    training_run_output_path,
):
    """
    train the model to two subsequent checkpoints (artificially close in time)
    and verify that:

    - the resulting parameters objects differ, implying training occurred
    """
    max_train_steps = 2000
    save_steps = int(max_train_steps / 2)
    initial_config_values = {
        "seed": 255,
        "dataset": {
            "train": {
                "paths": {
                    "dataset": os.path.join(
                        this_dir, "example-data/easy-task-train.csv"
                    )
                },
                "samples_before_next_set": None,
            },
            "validation": {
                "paths": {
                    "dataset": os.path.join(this_dir, "example-data/easy-task-val.csv"),
                }
            },
        },
        "tokenizer": {
            "vocab_path": os.path.join(this_dir, "example-data/easy-vocab.txt"),
            "vocab_size": 14,
            "pad_token_id": 2,
            "mask_token_id": 3,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "unk_token_id": 13,
            "other_special_token_ids": None,
            "max_length": 128,
        },
        "model": {
            "_name_": "AMPLIFY",
            "dropout_prob": 0.03,
            "embedding_init_range": 0.02,
            "decoder_init_range": 0.02,
            "norm_eps": 1e-05,
            "hidden_act": "gelu",
            "layer_norm_after_embedding": False,
            "layer_norm_before_last_layer": True,
            "rms_norm": False,
            "hidden_size": 320,
            "num_hidden_layers": 6,
            "num_attention_heads": 20,
            "intermediate_size": 1280,
            "ffn_bias": True,
            "att_bias": True,
        },
        "optimizer": {
            "_name_": "Adam",
            "lr": 4e-4,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "_name_": "LinearDecay",
            "final_ratio": 0.1,
            "final_step": 450000,
            "warmup_steps": 2000,
        },
        "trainer": {
            "dir": training_run_output_path,
            "resume": True,
            "max_steps": max_train_steps,
            "eval_steps": 10,
            "save_steps": max_train_steps,
            "gradient_clipping": None,
            "gradient_accumulation_steps": 1,
            "tf32": True,
            "disable_tqdm": True,
            "_name_": "MLM",
            "max_checkpoints": 10,
            "train": {
                "max_tokens": 64,
                "padding": "max_length",
                "pad_to_multiple_of": 8,
                "random_truncate": True,
                "mask_probability": 0.15,
                "num_workers": 4,
                "per_device_batch_size": 16,
                "label_smoothing": 0,
                "weights": {
                    "A": 1.0,
                    "C": 1.0,
                    "D": 1.0,
                    "E": 1.0,
                    "G": 1.0,
                    "S": 1.0,
                    "T": 1.0,
                    "V": 1.0,
                    "W": 1.0,
                },
                "exclude_special_tokens_replacement": False,
            },
            "validation": {
                "max_tokens": 64,
                "padding": "max_length",
                "pad_to_multiple_of": 8,
                "random_truncate": True,
                "mask_probability": 0.15,
                "num_workers": 4,
                "per_device_batch_size": 64,
                "label_smoothing": 0,
                "weights": {
                    "A": 1.0,
                    "C": 1.0,
                    "D": 1.0,
                    "E": 1.0,
                    "G": 1.0,
                    "S": 1.0,
                    "T": 1.0,
                    "V": 1.0,
                    "W": 1.0,
                },
                "exclude_special_tokens_replacement": False,
            },
        },
        "analysis": [
            {
                "device": 0,
                "from_checkpoint": None,
            },
            {
                "dataloader": {
                    "paths": [os.path.join(this_dir, "example-data/easy-task-val.csv")],
                    "max_tokens": 64,
                    "padding": "longest",
                    "pad_to_multiple_of": 8,
                    "random_truncate": False,
                    "num_workers": 0,
                    "per_device_batch_size": 64,
                }
            },
            {
                "umap_embedding_matrix": {
                    "n_neighbors": 20,
                    "min_dist": 0.1,
                    "n_epochs": 2000,
                    "low_memory": False,
                }
            },
            {
                "umap_proteins": {
                    "num_proteins": 10000,
                    "n_neighbors": 500,
                    "min_dist": 0.1,
                    "n_epochs": 2000,
                    "low_memory": False,
                }
            },
            {
                "mc_dropout": {
                    "num_proteins": 5,
                    "num_steps": 500,
                }
            },
            {
                "integrated_gradient": {
                    "num_proteins": 5,
                    "num_steps": 2000,
                    "batch_size": 32,
                }
            },
        ],
        "wandb": {
            "mode": "offline",
            "name": "unittest",
            "dir": os.path.join(training_run_output_path, "wandb"),
            "project": "test-project",
            "entity": None,
            "log_interval": int(max_train_steps / 5),
            "tags": [],
        },
    }
    config = OmegaConf.create(initial_config_values)

    # verify that there is no pre-existing checkpoint
    assert not os.path.exists(
        os.path.join(training_run_output_path, "checkpoints/checkpoint_1")
    )

    # train for a few steps
    # to generate a checkpoint
    config.trainer.resume = False
    trainer(config)

    # verify that a checkpoint was created
    assert os.path.exists(
        os.path.join(training_run_output_path, "checkpoints/checkpoint_1")
    )

    log_path = os.path.join(training_run_output_path, "wandb/wandb/metrics.json")
    with open(log_path, "r") as f:
        log_list = json.load(f)

    train_loss_series = np.array([float(x["train_loss"]) for x in log_list])

    # verify training loss decreases as steps progress, expressed as:
    # the logged sequence of training set losses
    # is "approximately monotonic", with no more than one consecutive increase,
    # and no increase greater than one percent of the loss value at that step
    for n, loss, change in zip(
        range(len(train_loss_series) - 1),
        train_loss_series[1:],
        np.diff(train_loss_series),
    ):
        assert (change < 0) or (
            1e-2 > (change / loss)
        ), f"Train loss increased beyond expected tolerance at index {n} in run"


def test__esm__saves_a_checkpoint_from_which_training_resumes(
    training_run_output_path,
):
    """
    train the model to two subsequent checkpoints (artificially close in time)
    and verify that:

    - the resulting parameters objects differ, implying training occurred
    """
    max_train_steps = 200
    save_steps = int(max_train_steps / 2)
    initial_config_values = {
        "seed": 255,
        "dataset": {
            "train": {
                "paths": {
                    "dataset": os.path.join(
                        this_dir, "example-data/easy-task-train.csv"
                    )
                },
                "samples_before_next_set": None,
            },
            "validation": {
                "paths": {
                    "dataset": os.path.join(this_dir, "example-data/easy-task-val.csv"),
                }
            },
        },
        "tokenizer": {
            "vocab_path": os.path.join(this_dir, "example-data/easy-vocab.txt"),
            "vocab_size": 14,
            "pad_token_id": 2,
            "mask_token_id": 3,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "unk_token_id": 13,
            "other_special_token_ids": None,
            "max_length": 128,
        },
        "model": {
            "_name_": "AMPLIFY",
            "dropout_prob": 0.03,
            "embedding_init_range": 0.02,
            "decoder_init_range": 0.02,
            "norm_eps": 1e-05,
            "hidden_act": "gelu",
            "layer_norm_after_embedding": False,
            "layer_norm_before_last_layer": True,
            "rms_norm": False,
            "hidden_size": 320,
            "num_hidden_layers": 6,
            "num_attention_heads": 20,
            "intermediate_size": 1280,
            "ffn_bias": True,
            "att_bias": True,
        },
        "optimizer": {
            "_name_": "Adam",
            "lr": 4e-4,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "_name_": "LinearDecay",
            "final_ratio": 0.1,
            "final_step": 450000,
            "warmup_steps": 2000,
        },
        "trainer": {
            "dir": training_run_output_path,
            "resume": True,
            "max_steps": max_train_steps,
            "eval_steps": 10,
            "save_steps": max_train_steps,
            "gradient_clipping": None,
            "gradient_accumulation_steps": 1,
            "tf32": True,
            "disable_tqdm": True,
            "_name_": "MLM",
            "max_checkpoints": 10,
            "train": {
                "max_tokens": 64,
                "padding": "max_length",
                "pad_to_multiple_of": 8,
                "random_truncate": True,
                "mask_probability": 0.15,
                "num_workers": 4,
                "per_device_batch_size": 16,
                "label_smoothing": 0,
                "weights": {
                    "A": 1.0,
                    "C": 1.0,
                    "D": 1.0,
                    "E": 1.0,
                    "G": 1.0,
                    "S": 1.0,
                    "T": 1.0,
                    "V": 1.0,
                    "W": 1.0,
                },
                "exclude_special_tokens_replacement": False,
            },
            "validation": {
                "max_tokens": 64,
                "padding": "max_length",
                "pad_to_multiple_of": 8,
                "random_truncate": True,
                "mask_probability": 0.15,
                "num_workers": 4,
                "per_device_batch_size": 64,
                "label_smoothing": 0,
                "weights": {
                    "A": 1.0,
                    "C": 1.0,
                    "D": 1.0,
                    "E": 1.0,
                    "G": 1.0,
                    "S": 1.0,
                    "T": 1.0,
                    "V": 1.0,
                    "W": 1.0,
                },
                "exclude_special_tokens_replacement": False,
            },
        },
        "analysis": [
            {
                "device": 0,
                "from_checkpoint": None,
            },
            {
                "dataloader": {
                    "paths": [os.path.join(this_dir, "example-data/easy-task-val.csv")],
                    "max_tokens": 64,
                    "padding": "longest",
                    "pad_to_multiple_of": 8,
                    "random_truncate": False,
                    "num_workers": 0,
                    "per_device_batch_size": 64,
                }
            },
            {
                "umap_embedding_matrix": {
                    "n_neighbors": 20,
                    "min_dist": 0.1,
                    "n_epochs": 2000,
                    "low_memory": False,
                }
            },
            {
                "umap_proteins": {
                    "num_proteins": 10000,
                    "n_neighbors": 500,
                    "min_dist": 0.1,
                    "n_epochs": 2000,
                    "low_memory": False,
                }
            },
            {
                "mc_dropout": {
                    "num_proteins": 5,
                    "num_steps": 500,
                }
            },
            {
                "integrated_gradient": {
                    "num_proteins": 5,
                    "num_steps": 2000,
                    "batch_size": 32,
                }
            },
        ],
        "wandb": {
            "mode": "offline",
            "name": "unittest",
            "dir": os.path.join(training_run_output_path, "wandb"),
            "project": "test-project",
            "entity": None,
            "log_interval": save_steps,
            "tags": [],
        },
    }
    config = OmegaConf.create(initial_config_values)

    # verify that there is no pre-existing checkpoint
    assert not os.path.exists(
        os.path.join(training_run_output_path, "checkpoints/checkpoint_1")
    )

    # train for a few steps
    # to generate a checkpoint
    config.trainer.resume = False
    trainer(config)

    # verify that a checkpoint was created
    assert os.path.exists(
        os.path.join(training_run_output_path, "checkpoints/checkpoint_1")
    )

    # and that it is the latest checkpoint so far
    assert not os.path.exists(
        os.path.join(training_run_output_path, "checkpoints/checkpoint_2")
    )

    # try to resume from that checkpoint
    # and train for ten more steps
    config.trainer.resume = True
    config.trainer.max_steps = config.trainer.max_steps + save_steps
    config.trainer.save_steps = save_steps
    trainer(config)

    # verify that new checkpoint was created
    assert os.path.exists(
        os.path.join(training_run_output_path, "checkpoints/checkpoint_2")
    )

    log_path = os.path.join(training_run_output_path, "wandb/wandb/metrics.json")
    with open(log_path, "r") as f:
        log_list = json.load(f)

    # configured log interval should have logged twice for first run,
    # one additional time for loading the saved model and continuing
    assert 3 == len(log_list)

    total_steps = int(log_list[-1]["num_steps"])
    assert config.trainer.max_steps == total_steps
