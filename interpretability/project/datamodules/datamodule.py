"""PyTorch Lightning data module for protein language model interpretability tasks.

This module provides data loading and preprocessing for various protein annotation tasks,
supporting both protein-level and amino-acid-level predictions, with automatic handling
of train/validation/test splits, label encoding, and data standardization.
"""
import multiprocessing
from pathlib import Path

import lightning
import torch
import polars as po
from torch.utils.data import DataLoader, Dataset

from project.utils.splitting import train_val_test_split
from project.utils.strs import SEED, PROTEIN_LENGTH_CUTOFF


class PolarsDataset(Dataset):
    """
    PyTorch Dataset wrapper for Polars DataFrames containing protein sequences and annotations.
    
    This dataset handles both protein-level and amino-acid-level tasks, automatically
    extracting sequences, IDs, and target labels from the DataFrame. For multiclass
    classification, it optionally maps original label values to contiguous class indices.
    
    Args:
        df: Polars DataFrame containing 'sequence' column and target columns
        label_mapping: Optional dictionary mapping original target values to class indices [0, num_classes-1].
                      Used for multiclass classification to ensure targets are in valid range.
    """
    def __init__(
        self,
        df,
        label_mapping: dict | None = None,
    ):
        self.df = df
        self.label_mapping = label_mapping

        if ('targets' in self.df.columns):
            self.target_cols = ['targets']
        else:
            self.target_cols = df.select(po.all().exclude(po.String)).columns


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        
        # Prepare the data dictionary
        data_item = {}
        
        if all(self.df.get_column(col).dtype == po.List for col in self.target_cols):
            # expand the list, 
            targets = torch.tensor([row[col] for col in self.target_cols], dtype=torch.long).view(-1)
            # Apply label mapping if provided (for multiclass classification)
            if self.label_mapping is not None:
                # Map each target value using the label mapping
                # Preserve padding values (-100) and other special values
                mapped_targets = []
                for val in targets:
                    val_int = int(val)
                    if val_int == -100:  # Padding value, keep as is
                        mapped_targets.append(-100)
                    elif val_int in self.label_mapping:
                        mapped_targets.append(self.label_mapping[val_int])
                    else:
                        # If value not in mapping, raise error to catch data issues
                        raise ValueError(f"Target value {val_int} not found in label_mapping. Available keys: {list(self.label_mapping.keys())}")
                targets = torch.tensor(mapped_targets, dtype=torch.long)
            data_item['targets'] = targets
        else:
            # Handle simple numerical targets and convert to a float32 tensor
            # Select numerical features, convert to float for compatibility with pytorch
            targets = torch.tensor([row[col] for col in self.target_cols], dtype=torch.float32)
            # Apply label mapping if provided (for protein_level multiclass classification)
            if self.label_mapping is not None and len(self.target_cols) == 1:
                # For protein-level multiclass, map the single target value
                val = int(targets.item())
                if val in self.label_mapping:
                    targets = torch.tensor([self.label_mapping[val]], dtype=torch.long)
                else:
                    raise ValueError(f"Target value {val} not found in label_mapping. Available keys: {list(self.label_mapping.keys())}")
            data_item['targets'] = targets

        # Handle the 'sequence' column separately
        if 'sequence' in row:
            data_item['sequence'] = row['sequence']

                # Handle the 'sequence' column separately
        if 'id' in row:
            data_item['id'] = row['id']
            
        return data_item


class PLMDataModule(lightning.LightningDataModule):
    """
    PyTorch Lightning DataModule for protein language model interpretability experiments.
    
    Handles loading, preprocessing, and splitting of protein sequence datasets with annotations.
    Supports multiple task types (binary/multiclass classification, regression) and task levels
    (protein-level or amino-acid-level predictions).
    
    Args:
        data_path: Path to parquet file containing the dataset
        task_level: "protein_level" or "amino_acid_level" - whether predictions are per-protein or per-residue
        task_type: "binary_classification", "multiclass_classification", or "regression"
        dataset_name: Name identifier for the dataset (for logging/tracking)
        dataset: Alternative name parameter (if provided, overrides dataset_name)
        batch_size: Batch size for DataLoaders
        train_size: Proportion of data for training set
        val_size: Proportion of data for validation set
        test_size: Proportion of data for test set
        expected_sequence_length: Maximum sequence length (sequences longer than this will raise an error)
        random_state: Random seed for dataset splitting
        num_workers: Number of worker processes for DataLoaders (None = CPU count - 1)
    """
    def __init__(
        self,
        data_path: str | Path,
        task_level: str = "protein_level",  # or "amino_acid_level"
        task_type: str = "binary_classification",  # or "regression"
        dataset_name:str | None = "secondary_structure",
        dataset:str | None = None,
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        expected_sequence_length: int = PROTEIN_LENGTH_CUTOFF,
        random_state: int = SEED,
        num_workers: int | None = multiprocessing.cpu_count() - 1,
    ):
        super().__init__()
        self.data_path = data_path
        self.task_level = task_level # 'protein_level' or 'amino_acid_level'
        self.task_type = task_type # 'binary_classification', 'multiclass_classification', or 'regression'
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.expected_sequence_length = expected_sequence_length
        self.num_workers = num_workers
        if dataset:
            self.dataset_name = dataset
        self.dataset_name = dataset_name

        # Will be set in setup()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = None
        self.target_cols = None
        self.target_classes = None
        self.label_mapping = None  # Maps original target values to class indices [0, num_classes-1]

    def setup(self, stage: str | None = None):
        df = po.read_parquet(self.data_path)
        # Assert that there is a 'sequence' column
        assert "sequence" in df.columns, "DataFrame must contain a 'sequence' column."
        # Assert that all sequences are of the expected length
        assert df["sequence"].str.len_chars().max() <= self.expected_sequence_length, f"All sequences must be <= {self.expected_sequence_length}."

        if self.task_level == 'amino_acid_level':
            self.target_cols = ['targets']
            if 'multiclass_classification' in self.task_type:
                self.target_classes = df['targets'].list.explode().unique().sort() # Number of unique classes, sorted
                self.num_targets = len(self.target_classes) # Number of unique targets
                # Create label mapping: map original target values to class indices [0, num_classes-1]
                self.label_mapping = {int(orig_val): idx for idx, orig_val in enumerate(self.target_classes)}
            elif 'binary_classification' in self.task_type:
                self.target_classes = [0,1]
                self.num_targets = 1
                self.label_mapping = None
        elif self.task_level == 'protein_level':
            # Match PolarsDataset logic: check for 'targets' column first
            if 'targets' in df.columns:
                self.target_cols = ['targets']
            else:
                # All non-string columns are targets (excludes 'id', 'sequence' which are strings)
                self.target_cols = df.select(po.all().exclude(po.String)).columns
            
            if self.task_type == 'multiclass_classification':
                # For multiclass classification, count unique classes
                # If there's a single 'targets' column, count unique values in that column
                # If multiple columns (one-hot encoded), number of classes = number of columns
                if len(self.target_cols) == 1 and 'targets' in self.target_cols:
                    self.target_classes = df[self.target_cols[0]].unique().sort()
                    self.num_targets = len(self.target_classes)
                    # Create label mapping: map original target values to class indices [0, num_classes-1]
                    self.label_mapping = {int(orig_val): idx for idx, orig_val in enumerate(self.target_classes)}
                else:
                    # One-hot encoded: number of classes = number of columns
                    self.num_targets = len(self.target_cols)
                    self.label_mapping = None
            else:
                self.num_targets = len(self.target_cols)  # Non-string columns are targets
  
        # Split the dataset into train, validation, and test sets if we've provided one
        if "split" in df.columns:
            # Splits are predefined, pull them from the 'split' column
            self.train_data, self.val_data, self.test_data = (df.filter(po.col('split') == 'train').drop('split'), 
                                                            df.filter(po.col('split') == 'val').drop('split'), 
                                                            df.filter(po.col('split') == 'test').drop('split'))
        else:
            # Otherwise, make our own random splits
            self.train_data, self.val_data, self.test_data = train_val_test_split(
                df=df,
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                seed=SEED,
            )

        # If we're doing regression, standardize
        if self.task_type == 'regression':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Use train data to fit the scaler
            scaler.fit(self.train_data.to_pandas()[self.target_cols])
            for _data in [self.train_data, self.val_data, self.test_data]:
                # Scale the data based on train data
                transformed_data = scaler.transform(_data.to_pandas()[self.target_cols])
                # replace each column with its transformed version
                for i, col in enumerate(self.target_cols):
                    _data.replace_column((len(_data.columns) - len(self.target_cols) + i), po.Series(col, transformed_data[:,i]))

            self.scaler = scaler

    def _collate_fn(self, batch):
        # Get the targets from the batch - they'll be tensors of different lengths
        targets = [i['targets'] for i in batch]
        # Pad each tensor 
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
        batch_dict = {
            'sequence': [i['sequence'] for i in batch],
            'id': [i['id'] for i in batch],
            'targets': targets,
        }
        return batch_dict
        

    def train_dataloader(self):
        return DataLoader(
            PolarsDataset(self.train_data, label_mapping=self.label_mapping),
            collate_fn = self._collate_fn if self.task_level == 'amino_acid_level' else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            PolarsDataset(self.val_data, label_mapping=self.label_mapping),
            collate_fn = self._collate_fn if self.task_level == 'amino_acid_level' else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            PolarsDataset(self.test_data, label_mapping=self.label_mapping),
            collate_fn = self._collate_fn if self.task_level == 'amino_acid_level' else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
