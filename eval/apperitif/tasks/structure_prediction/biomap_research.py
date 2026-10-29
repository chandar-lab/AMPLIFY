from apperitif.tasks.structure_prediction.StructurePredictionTask import (
    StructurePredictionTask,
    StructurePredictionDataConfig,
)
from dataclasses import dataclass
from datasets import load_dataset
from typing import Optional


@dataclass
class BiomapResearchStructurePredictionConfig(StructurePredictionDataConfig):
    """
    Configuration for the Biomap Research structure prediction task.
    Uses a HF dataset name and enforces standardized column names.
    """

    task_type: str = "biomap_research_structure_prediction"
    dataset_name: str = "biomap-research/contact_prediction_binary"
    split_train: bool = False
    split_seed: Optional[int] = 42


class BiomapResearchStructurePredictionTask(StructurePredictionTask):
    @property
    def name(self):
        return "biomap_research_structure_prediction"

    def get_data(self):
        dataset = load_dataset(self.config.data_config.dataset_name)
        # Optionally split train into train/valid
        if self.config.data_config.split_train:
            train_dataset = dataset["train"].train_test_split(test_size=0.1, seed=self.config.data_config.split_seed)
            dataset["valid"] = train_dataset["test"]
            dataset["train"] = train_dataset["train"]
        # Minimal pattern: expect a 'valid' split and expose it as 'validation'
        dataset["validation"] = dataset.pop("valid")  # Will fail if no valid split was created
        return dataset
