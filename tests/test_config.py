import os

from omegaconf import OmegaConf, DictConfig

from amplify.trainer import trainer
from amplify.config import ConfigError, ConfigValidator
from amplify.config.validator import (
    RequireMax,
    RequireMin,
    RequireInRange,
    RequireInSet,
)

from .fixtures import training_run_output_path

this_dir = os.path.abspath(os.path.dirname(__file__))


def test__require_in_range__returns_true_if_in_range():
    is_in_range = RequireInRange(0, 1, include_lower=True, include_upper=True)

    assert is_in_range(0.2)
    assert not is_in_range(2.7)
    assert not is_in_range(-3)


def test__require_in_range__includes_endpoints_by_default():
    is_in_range = RequireInRange(0, 1)

    assert is_in_range(0.0)
    assert is_in_range(1.0)


def require_in_range_excludes_left_endpoint_when_specified():
    is_in_range = RequireInRange(0, 1, include_lower=False)

    assert is_in_range(0.2)
    assert not is_in_range(1.1)
    assert not is_in_range(0.0)


def require_in_range_excludes_right_endpoint_when_specified():
    is_in_range = RequireInRange(0, 1, include_upper=False)

    assert is_in_range(0.2)
    assert not is_in_range(1.1)
    assert not is_in_range(1.0)


def test__require_in_set__returns_true_if_in_range_for_a_specified_list():
    is_in_set = RequireInSet(["abc", "def", "q", "wxy"])

    assert is_in_set("abc")
    assert is_in_set("def")
    assert is_in_set("q")
    assert is_in_set("wxy")
    assert not is_in_set("c")
    assert not is_in_set("z")


def test__require_in_set__returns_true_if_in_range_for_a_specified_set():
    is_in_set = RequireInSet({"abc", "def", "q", "wxy"})

    assert is_in_set("abc")
    assert is_in_set("def")
    assert is_in_set("q")
    assert is_in_set("wxy")
    assert not is_in_set("c")
    assert not is_in_set("z")


def test__require_in_set__optionally_ignores_casing_of_strings():
    is_in_set = RequireInSet({"abc", "def", "q", "wxy"}, ignore_case=True)

    assert is_in_set("abc")
    assert is_in_set("def")
    assert is_in_set("q")
    assert is_in_set("Q")
    assert is_in_set("wxy")
    assert is_in_set("wXy")
    assert is_in_set("WXY")
    assert is_in_set("wXY")
    assert not is_in_set("C")
    assert not is_in_set("z")


def test__require_in_set__requires_casing_of_strings_when_specified():
    is_in_set = RequireInSet({"abc", "def", "q", "wxy"}, ignore_case=False)

    assert is_in_set("abc")
    assert is_in_set("def")
    assert is_in_set("q")
    assert not is_in_set("Q")
    assert is_in_set("wxy")
    assert not is_in_set("wXy")
    assert not is_in_set("WXY")
    assert not is_in_set("wXY")
    assert not is_in_set("C")
    assert not is_in_set("z")


def test__config_validator__returns_ok_if_all_functions_in_flat_config_return_true():

    def _field0_validator(x):
        return 0 < x and x < 1

    def _field1_validator(x):
        return x in "abc"

    def _field2_validator(x):
        return len(x) < 5 and 0 < len(x) and x[1] == "2"

    validator = ConfigValidator(
        {
            "field0": _field0_validator,
            "field1": _field1_validator,
            "field2": _field2_validator,
        }
    )

    assert validator.validate(
        {
            "field0": 0.8,
            "field1": "c",
            "field2": "1234",
        }
    ).is_ok()


def test__config_validator__returns_config_error_if_one_function_in_flat_config_returns_false():

    def _field0_validator(x):
        return 0 < x and x < 1

    def _field1_validator(x):
        return x in "abc"

    def _field2_validator(x):
        return len(x) < 5 and 0 < len(x) and x[1] == "2"

    validator = ConfigValidator(
        {
            "field0": _field0_validator,
            "field1": _field1_validator,
            "field2": _field2_validator,
        }
    )

    assert validator.validate(
        {
            "field0": 0.8,
            "field1": "c",
            "field2": "2357",  # fails
        }
    ).is_not_ok()


def test__config_validator__on_error__returns_listing_of_all_failed_fields():

    def _field0_validator(x):
        return 0 < x and x < 1

    def _field1_validator(x):
        return x in "abc"

    def _field2_validator(x):
        return len(x) < 5 and 0 < len(x) and x[1] == "2"

    validator = ConfigValidator(
        {
            "field0": _field0_validator,
            "field1": _field1_validator,
            "field2": _field2_validator,
        }
    )

    failures = validator.validate(
        {
            "field0": 2.7,  # fails
            "field1": "c",
            "field2": "2357",  # fails
        }
    )

    assert "field0" in failures
    assert "field2" in failures


def test__config_validator__returns_ok_if_no_failures_in_nested_config():

    def _field0_validator(x):
        return 0 < x and x < 1

    def _field1_validator(x):
        return x in "abc"

    def _field2_validator(x):
        return len(x) < 5 and 0 < len(x) and x[1] == "2"

    validator = ConfigValidator(
        {
            "group0": {
                "field0": _field0_validator,
                "field1": _field1_validator,
            },
            "group1": {
                "field2": _field2_validator,
            },
        }
    )

    assert validator.validate(
        {
            "field0": 0.8,
            "field1": "c",
            "field2": "1234",
        }
    ).is_ok()


def test__config_validator__returns_error_result_for_errors_in_nested_config():

    def _field0_validator(x):
        return 0 < x and x < 1

    def _field1_validator(x):
        return x in "abc"

    def _field2_validator(x):
        return len(x) < 5 and 0 < len(x) and x[1] == "2"

    def _field3_validator(x):
        return x is not None

    def _field4_validator(x):
        return x in [5, 7, 11]

    validator = ConfigValidator(
        {
            "group0": {
                "field0": _field0_validator,
                "field1": _field1_validator,
            },
            "group1": {
                "field2": _field2_validator,
                "group2": {
                    "field3": _field3_validator,
                    "field4": _field4_validator,
                },
            },
        }
    )

    failures = validator.validate(
        DictConfig(
            {
                "group0": {
                    "field0": 2.7,  # fails
                    "field1": "c",
                },
                "group1": {
                    "field2": "2357",  # fails
                    "group2": {
                        "field3": None,  # fails
                        "field4": 7,
                    },
                },
            }
        )
    )

    assert "group0.field0" in failures
    assert "group1.field2" in failures
    assert "group1.group2.field3" in failures


def test__config_validator__returns_failure_reasons_when_available():

    validator = ConfigValidator(
        {
            "group0": {
                "field0": RequireInRange(0, 1),
                "field1": RequireInSet(["a", "b", "c"]),
            },
            "group1": {
                "field2": RequireMin(0),
                "group2": {
                    "field3": RequireMax(1024),
                    "field4": RequireInSet(["ok", "wow"]),
                    "field5": RequireInSet(["ok", "wow"]),
                },
            },
        }
    )

    failures = validator.validate(
        DictConfig(
            {
                "group0": {
                    "field0": 2.7,  # fails
                    "field1": "c",
                },
                "group1": {
                    "field2": -1,  # fails
                    "group2": {
                        "field3": 2048,  # fails
                        "field4": "neat",  # fails
                        "field5": "ok",
                        "field6": 512,  # not checked
                    },
                },
                "field7": "wow",  # not checked
            }
        )
    )

    assert "field1" not in failures
    assert "field6" not in failures
    assert "field7" not in failures

    assert "group0.field0" in failures
    assert "Value must be between 0 and 1" == failures["group0.field0"]

    assert "group1.field2" in failures
    assert "Value must be greater than 0" == failures["group1.field2"]

    assert "group1.group2.field3" in failures
    assert "Value must be less than 1024" == failures["group1.group2.field3"]

    assert "group1.group2.field4" in failures
    assert "Value must be one of: ['ok', 'wow']" == failures["group1.group2.field4"]


def test__config_validator__works_with_validation_classes_or_tuple_of_callable_and_format_str():

    def _require_in_range_0_1(x):
        return 0 < x and x < 1

    def _require_in_a_b_c(x):
        return x in "abc"

    validator = ConfigValidator(
        {
            "group0": {
                "field0a": (_require_in_range_0_1, "Must be between 0 and 1"),
                "field1a": _require_in_a_b_c,
            },
            "group1": {
                "group2": {
                    "field5a": (_require_in_a_b_c, "Must be one of: ['a', 'b', 'c']"),
                },
            },
        }
    )

    failures = validator.validate(
        DictConfig(
            {
                "group0": {
                    "field0a": 2.7,  # fails
                    "field1a": "d",  # fails
                },
                "group1": {
                    "field2": -1,  # fails
                    "group2": {
                        "field3": 2048,  # fails
                        "field4": "neat",  # fails
                        "field5a": "ok",
                        "field6": 512,  # not checked
                    },
                },
                "field7": "wow",  # not checked
            }
        )
    )

    assert "group0.field0a" in failures
    assert "Must be between 0 and 1" == failures["group0.field0a"]

    assert "group0.field1a" in failures
    assert "Invalid specification: d" == failures["group0.field1a"]

    assert "group1.group2.field5a" in failures
    assert "Must be one of: ['a', 'b', 'c']" == failures["group1.group2.field5a"]


def test__config_validator__raises_typeerror_if_schema_not_specified_with_callables_or_tuples():

    try:
        validator = ConfigValidator(
            {
                "group0": {
                    "field0a": RequireInRange(0, 1),
                },
                "group1": {
                    "group2": {
                        "malformed_field": "not valid",
                    },
                },
            }
        )
        assert False, "expected error raised"
    except TypeError as e:
        expected = f"In ConfigValidator, at key group1.group2.malformed_field: each leaf node of the schema must be a callable, or a tuple consisting of a callable and an error string."
        assert expected == str(e)


def test__config_validator__raises_error_from_trainer_if_config_misspecified(
    training_run_output_path,
):
    """
    verify that training is immediately stopped by an error if a config value is incorrect
    """
    max_train_steps = 10
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
            "max_length": 2147483647,  # should fail
        },
        "model": {
            "_name_": "ESM",
            "dropout_prob": 1.3,  # should fail
            "embedding_init_range": 0.02,
            "decoder_init_range": 0.02,
            "norm_eps": 1e-05,
            "hidden_act": "not-a-real-activation",  # should fail with distinctive error on start
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
            "lr": 4e-3,
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
                "mask_probability": -0.15,  # should fail
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
                "mask_probability": 1.15,  # should fail
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

    try:
        trainer(config)
        assert False, "expected error raised"
    except ConfigError as e:
        assert "tokenizer.max_length" in e.details
        assert "model.hidden_act" in e.details
        assert "model.dropout_prob" in e.details
        assert "trainer.train.mask_probability" in e.details
        assert "trainer.validation.mask_probability" in e.details
