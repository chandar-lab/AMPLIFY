"""
Utility functions for handling Hydra config mapping.
These functions help map anonymous config objects to specific config classes
based on discriminator fields like model_type, evaluator_type, and task_type.
"""

from typing import Any, Dict, Union
from omegaconf import DictConfig


def resolve_evaluator_target(
    task_type: str = "sequence_classification",
    head_type: str = "linear_probing",
) -> str:
    """
    Resolve the _target_ field for the evaluator based on task and head type.
    Args:
        task_type: The type of task (e.g., "sequence_classification", "token_classification", etc.)
        head_type: The type of head to use (e.g., "linear_probing", "knn", etc.)
    Returns:
        The fully qualified class path for the evaluator.
    """
    # Mapping of task types to evaluator classes
    task_evaluator_mapping = {
        "sequence_classification": {
            "linear_probing": "apperitif.evaluation.SequenceClassificationEvaluator.LinearProbingSequenceClassificationEvaluatorConfig",
            "knn": "apperitif.evaluation.SequenceClassificationEvaluator.KNNSequenceClassificationEvaluatorConfig",
            "mlp": "apperitif.evaluation.SequenceClassificationEvaluator.MLPSequenceClassificationEvaluatorConfig",
        },
        "sequence_regression": {
            "linear_probing": "apperitif.evaluation.SequenceRegressionEvaluator.LinearProbingSequenceRegressionEvaluatorConfig",
            "knn": "apperitif.evaluation.SequenceRegressionEvaluator.KNNSequenceRegressionEvaluatorConfig",
            "mlp": "apperitif.evaluation.SequenceRegressionEvaluator.MLPSequenceRegressionEvaluatorConfig",
        },
        "token_classification": {
            "mlp": "apperitif.evaluation.TokenClassificationEvaluator.MLPTokenClassificationEvaluatorConfig",
        },
        "sequence_structure": {
            "mlp": "apperitif.evaluation.SequenceStructureEvaluator.SequenceStructureEvaluatorConfig",
        },
    }

    # Resolve the evaluator class based on task and head type
    if task_type not in task_evaluator_mapping:
        raise ValueError(f"Unknown task_type: {task_type}. Available types: {list(task_evaluator_mapping.keys())}")
    if head_type not in task_evaluator_mapping[task_type]:
        raise ValueError(
            f"Unknown head_type '{head_type}' for task '{task_type}'. "
            f"Available: {list(task_evaluator_mapping[task_type].keys())}"
        )
    evaluator_class_name = task_evaluator_mapping[task_type][head_type]
    return evaluator_class_name


# # Import all model configs
# from apperitif.models.ESM import ESMConfig
# from apperitif.models.Amplify import AmplifyConfig

# # Import all evaluator configs
# from apperitif.evaluation.SequenceClassificationEvaluator import (
#     KNNSequenceClassificationEvaluatorConfig,
#     LinearProbingSequenceClassificationEvaluatorConfig,
#     MLPSequenceClassificationEvaluatorConfig,
# )
# from apperitif.evaluation.SequenceRegressionEvaluator import (
#     KNNSequenceRegressionEvaluatorConfig,
#     LinearProbingSequenceRegressionEvaluatorConfig,
#     MLPSequenceRegressionEvaluatorConfig,
# )
# from apperitif.evaluation.TokenClassificationEvaluator import (
#     MLPTokenClassificationEvaluatorConfig,
# )
# from apperitif.evaluation.SequenceStructureEvaluator import (
#     SequenceStructureEvaluatorConfig,
# )

# # Import all task configs
# from apperitif.tasks.sequence_classification.SequenceClassificationTask import SequenceClassificationDataConfig
# from apperitif.tasks.sequence_regression.SequenceRegressionTask import RegressionDataConfig
# from apperitif.tasks.token_classification.TokenClassificationTask import TokenClassificationDataConfig
# from apperitif.tasks.sequence_classification.biomap_research import BiomapResearchSequenceClassificationConfig
# from apperitif.tasks.sequence_regression.biomap_research import BiomapResearchSequenceRegressionConfig
# from apperitif.tasks.token_classification.biomap_research import BiomapResearchTokenClassificationConfig


# # Model type mapping - simplified to base model types
# MODEL_CONFIG_MAPPING = {
#     # ESM models - the task will determine which wrapper to use
#     "esm": {
#         "sequence": ESMConfig,
#         "token": ESMConfig,
#         "structure": ESMConfig,
#     },
#     # Amplify models - the task will determine which wrapper to use
#     "amplify": {
#         "sequence": AmplifyConfig,
#         "token": AmplifyConfig,
#         "structure": AmplifyConfig,
#     },
# }

# # Evaluator type mapping
# EVALUATOR_CONFIG_MAPPING = {
#     # Sequence classification evaluators
#     "knn_sequence_classification": KNNSequenceClassificationEvaluatorConfig,
#     "linear_probing_sequence_classification": LinearProbingSequenceClassificationEvaluatorConfig,
#     "mlp_sequence_classification": MLPSequenceClassificationEvaluatorConfig,
#     # Sequence regression evaluators
#     "knn_sequence_regression": KNNSequenceRegressionEvaluatorConfig,
#     "linear_probing_sequence_regression": LinearProbingSequenceRegressionEvaluatorConfig,
#     "mlp_sequence_regression": MLPSequenceRegressionEvaluatorConfig,
#     # Token classification evaluators
#     "mlp_token_classification": MLPTokenClassificationEvaluatorConfig,
#     # Structure evaluators
#     "sequence_structure": SequenceStructureEvaluatorConfig,
# }

# # Task type mapping
# TASK_CONFIG_MAPPING = {
#     # Base task configs
#     "sequence_classification": SequenceClassificationDataConfig,
#     "sequence_regression": RegressionDataConfig,
#     "token_classification": TokenClassificationDataConfig,
#     # Specific task configs
#     "biomap_research_sequence_classification": BiomapResearchSequenceClassificationConfig,
#     "biomap_research_sequence_regression": BiomapResearchSequenceRegressionConfig,
#     "biomap_research_token_classification": BiomapResearchTokenClassificationConfig,
# }


# def resolve_model_config(config: Union[Dict[str, Any], DictConfig], task_encoder_type: str = "sequence") -> Any:
#     """
#     Resolve an anonymous model config to the appropriate config class.

#     Args:
#         config: Anonymous config dict/DictConfig with model_type field
#         task_encoder_type: The type of encoder needed ("sequence", "token", "structure")

#     Returns:
#         Properly typed config object

#     Raises:
#         ValueError: If model_type is not recognized
#     """
#     if isinstance(config, DictConfig):
#         config = dict(config)

#     model_type = config.get("model_type")
#     if model_type not in MODEL_CONFIG_MAPPING:
#         raise ValueError(f"Unknown model_type: {model_type}. Available types: {list(MODEL_CONFIG_MAPPING.keys())}")

#     encoder_configs = MODEL_CONFIG_MAPPING[model_type]
#     if task_encoder_type not in encoder_configs:
#         raise ValueError(
#             f"Unknown encoder type '{task_encoder_type}' for model '{model_type}'. Available: {list(encoder_configs.keys())}"
#         )

#     config_class = encoder_configs[task_encoder_type]
#     return config_class(**config)


# def resolve_evaluator_config(config: Union[Dict[str, Any], DictConfig]) -> Any:
#     """
#     Resolve an anonymous evaluator config to the appropriate config class.

#     Args:
#         config: Anonymous config dict/DictConfig with evaluator_type field

#     Returns:
#         Properly typed config object

#     Raises:
#         ValueError: If evaluator_type is not recognized
#     """
#     if isinstance(config, DictConfig):
#         config = dict(config)

#     evaluator_type = config.get("evaluator_type")
#     if evaluator_type not in EVALUATOR_CONFIG_MAPPING:
#         raise ValueError(
#             f"Unknown evaluator_type: {evaluator_type}. Available types: {list(EVALUATOR_CONFIG_MAPPING.keys())}"
#         )

#     config_class = EVALUATOR_CONFIG_MAPPING[evaluator_type]
#     return config_class(**config)


# def resolve_task_config(config: Union[Dict[str, Any], DictConfig]) -> Any:
#     """
#     Resolve an anonymous task config to the appropriate config class.

#     Args:
#         config: Anonymous config dict/DictConfig with task_type field

#     Returns:
#         Properly typed config object

#     Raises:
#         ValueError: If task_type is not recognized
#     """
#     if isinstance(config, DictConfig):
#         config = dict(config)

#     task_type = config.get("task_type")
#     if task_type not in TASK_CONFIG_MAPPING:
#         raise ValueError(f"Unknown task_type: {task_type}. Available types: {list(TASK_CONFIG_MAPPING.keys())}")

#     config_class = TASK_CONFIG_MAPPING[task_type]
#     return config_class(**config)


# def resolve_all_configs(
#     model_config: Union[Dict[str, Any], DictConfig],
#     evaluator_config: Union[Dict[str, Any], DictConfig],
#     data_config: Union[Dict[str, Any], DictConfig],
# ) -> tuple:
#     """
#     Resolve all configs at once for convenience.

#     Args:
#         model_config: Anonymous model config
#         evaluator_config: Anonymous evaluator config
#         data_config: Anonymous data/task config

#     Returns:
#         Tuple of (resolved_model_config, resolved_evaluator_config, resolved_data_config)
#     """
#     # First resolve data config to determine task type
#     resolved_data_config = resolve_task_config(data_config)

#     # Determine encoder type from task type
#     task_type = resolved_data_config.task_type
#     if "sequence_classification" in task_type or "sequence_regression" in task_type:
#         encoder_type = "sequence"
#     elif "token_classification" in task_type:
#         encoder_type = "token"
#     elif "structure" in task_type:
#         encoder_type = "structure"
#     else:
#         # Default to sequence for unknown task types
#         encoder_type = "sequence"

#     # Resolve model config with the determined encoder type
#     resolved_model_config = resolve_model_config(model_config, encoder_type)
#     resolved_evaluator_config = resolve_evaluator_config(evaluator_config)

#     return (
#         resolved_model_config,
#         resolved_evaluator_config,
#         resolved_data_config,
#     )
