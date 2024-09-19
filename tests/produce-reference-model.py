import os
import sys

from omegaconf import OmegaConf
import torch

src_dir = os.path.abspath(os.path.join(".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from amplify.trainer import trainer

this_dir = os.path.dirname(__file__)

output_path = os.path.join(this_dir, "reference-model")

max_train_steps = 100000
initial_config_values = {
    "seed": 255,
    "dataset": {
        "train": {
            "paths": {
                "dataset": os.path.join(this_dir, "example-data/big-easy-train.csv")
            },
            "samples_before_next_set": None,
        },
        "validation": {
            "paths": {
                "dataset": os.path.join(this_dir, "example-data/big-easy-val.csv"),
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
        "dir": output_path,
        "resume": True,
        "max_steps": max_train_steps,
        "eval_steps": 10,
        "save_steps": 5000,
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
                "paths": [os.path.join(this_dir, "example-data/big-easy-val.csv")],
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
        "dir": os.path.join(output_path, "wandb"),
        "project": "test-project",
        "entity": None,
        "log_interval": 100,
        "tags": [],
    },
}
# config = AMPLIFYConfig(OmegaConf.create(initial_config_values))
config = OmegaConf.create(initial_config_values)

config_path = os.path.join(output_path, "config.yaml")
with open(config_path, "w") as f:
    OmegaConf.save(config, f)

# verify that there is no pre-existing checkpoint
assert not os.path.exists(os.path.join(output_path, "checkpoints/checkpoint_1"))

# train for a few steps
# to generate a checkpoint
trainer(config)
