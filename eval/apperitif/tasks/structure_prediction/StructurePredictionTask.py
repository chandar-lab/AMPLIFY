from apperitif.tasks.AbsTask import AbsTask, TaskConfig, DataConfig
from apperitif.evaluation.SequenceStructureEvaluator import (
    SequenceStructureEvaluator,
    SequenceStructureEvaluatorConfig,
)
from dataclasses import dataclass
from apperitif.models.Amplify import AmplifyStructureEncoder, AmplifyConfig
from apperitif.models.ESM import ESMSequenceEncoder, ESMConfig
from apperitif.models.Encoder import EncoderConfig
from apperitif.models.TRM import TRMStructureEncoder, TRMConfig
from apperitif.utils.evaluator_utils import EvaluatorResult
from transformers import AutoTokenizer
from typing import Dict, Any, Tuple, Union


@dataclass
class StructurePredictionDataConfig(DataConfig):
    """
    Configuration for structure prediction datasets.
    This can include paths, preprocessing options, etc.
    """

    task_type: str = "sequence_structure"  # Discriminator field for Hydra
    sequence_column: str = "seq"  # Column name for sequences
    label_column: str = "label"  # Column name for labels / contact maps
    label_dtype: str = "int"  # Data type for labels (e.g., "int")


class StructurePredictionTask(AbsTask):
    """
    Abstract task for structure prediction using structured config objects.
    """

    @classmethod
    def from_config(cls, config: TaskConfig):
        """
        Factory method to create a StructurePredictionTask instance from a TaskConfig object.

        Args:
            config: TaskConfig object containing all necessary configurations

        Returns:
            StructurePredictionTask instance with the provided configuration
        """
        # For now, structure prediction only has the base class
        # Future extensions can add specific implementations here

        from apperitif.tasks.structure_prediction.biomap_research import (
            BiomapResearchStructurePredictionTask,
            BiomapResearchStructurePredictionConfig,
        )

        data_config = config.data_config

        # Map data config classes to task classes for structure prediction
        config_to_task_mapping = {
            StructurePredictionDataConfig: cls,  # Base class
            BiomapResearchStructurePredictionConfig: BiomapResearchStructurePredictionTask,
        }

        # Find the appropriate task class based on data_config type
        data_config_type = type(data_config)

        if data_config_type in config_to_task_mapping:
            task_class = config_to_task_mapping[data_config_type]
            return task_class(config)
        else:
            return cls(config)

    @property
    def name(self):
        return "structure_prediction"

    def get_data(self):
        """To be implemented in child classes: returns datasets."""
        raise NotImplementedError

    def get_encoder(self) -> Tuple[Any, Any]:
        """
        Get encoder and collate function from the task's model config.

        Returns:
            Tuple of (encoder, collate_fn)
        """
        model_config = self.config.model_config

        data_config = self.config.data_config

        # Ensure we have a structured config object
        if not isinstance(model_config, (ESMConfig, AmplifyConfig, TRMConfig, EncoderConfig)):
            raise ValueError(f"Expected structured config object, got {type(model_config)}")

        # Create encoder based on config type and model_type
        if isinstance(model_config, AmplifyConfig):
            encoder = AmplifyStructureEncoder(config=model_config)
        elif isinstance(model_config, ESMConfig):
            encoder = ESMSequenceEncoder(config=model_config)
        elif isinstance(model_config, TRMConfig):
            encoder = TRMStructureEncoder(config=model_config)
        else:
            if model_config.model_type.lower() == "amplify":
                encoder = AmplifyStructureEncoder(config=model_config)
            elif model_config.model_type.lower() == "esm":
                encoder = ESMSequenceEncoder(config=model_config)
            elif model_config.model_type.lower() == "trm":
                encoder = TRMStructureEncoder(config=model_config)
            else:
                raise ValueError(f"Unknown model_type: {model_config.model_type}")

        # Create collate function with reasonable defaults, following other task patterns
        collate_fn = encoder.get_collate_fn(
            sequence_column=getattr(data_config, "sequence_column", "seq"),
            label_column=getattr(data_config, "label_column", "labels"),
            device=self.device,
        )

        return encoder, collate_fn

    def get_evaluator(self, datasets: Dict[str, Any], collate_fn: Any) -> Any:
        """
        Get evaluator from the task's evaluator config.

        Args:
            datasets: Dictionary of datasets (train, validation, test)
            collate_fn: Collate function for the datasets

        Returns:
            Configured evaluator instance
        """
        evaluator_config = self.config.evaluator_config
        # Directly use SequenceStructureEvaluator (no head_type needed for now)
        if not isinstance(evaluator_config, SequenceStructureEvaluatorConfig):
            evaluator_config = SequenceStructureEvaluatorConfig(**evaluator_config)
        return SequenceStructureEvaluator(config=evaluator_config, datasets=datasets, collate_fn=collate_fn)

    def evaluate(self, **kwargs) -> EvaluatorResult:
        """
        Evaluate the model for this task.

        Args:
            **kwargs: Additional arguments passed to the evaluator

        Returns:
            EvaluatorResult: Standardized evaluation results with hyperparameter information
        """
        # Get datasets
        datasets = self.get_data()

        # Get encoder and collate function
        encoder, collate_fn = self.get_encoder()

        encoder = encoder.to(self.device)

        # Get evaluator
        evaluator = self.get_evaluator(datasets, collate_fn)

        # Merge config options with kwargs
        eval_kwargs = {
            "return_all_scores": self.config.return_all_results,
            "do_test": self.config.do_test,
            "verbose": self.config.verbose,  # Default verbose setting
            "task_name": self.config.name,
            "task_description": self.config.description,
            "output_folder": self.config.output_folder,
            **kwargs,
        }

        # Run evaluation
        evaluator(encoder, **eval_kwargs)

        # Save results to JSON if output folder is configured
        if self.config.output_folder:
            self.aggregate_results_in_json()
