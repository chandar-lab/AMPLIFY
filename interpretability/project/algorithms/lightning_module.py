"""PyTorch Lightning module for training linear probes on protein language model embeddings.

This module implements the training and evaluation logic for linear probing experiments,
where simple linear classifiers/regressors are trained on frozen PLM embeddings to assess
what information is encoded at different layers. Includes support for both trained and
naive (randomly initialized) control probes.
"""
import dataclasses
from dataclasses import dataclass
from logging import getLogger
from typing import Any
from pathlib import Path

import hydra_zen
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW

from project.algorithms.networks.linear import LinearProbe
from project.algorithms.networks.protein_language_model import ProteinLanguageModel
from project.utils.strs import SEED, linear_probe_results_dir
from project.utils.functions import scramble_sequences, shuffle_tensor, protein_analysis, reset_to_pytorch_defaults
from project.utils.metrics import get_metrics_dict, compute_metric

logger = getLogger(__name__)

# --- Configuration Dataclasses (Kept as is) ---

@hydra_zen.hydrated_dataclass(
    target=ProteinLanguageModel,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class ProteinLanguageModelConfig:
    model_name: str
    layer_to_use: int
    normalize_embeddings:bool


@hydra_zen.hydrated_dataclass(
    target=LinearProbe,
    frozen=False,
    unsafe_hash=True,
    populate_full_signature=True,
)
class LinearModelConfig:
    """Configuration for the Linear probe."""
    input_dim: int | None = None
    output_dim: int | None = None
    purpose: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    adam_betas: tuple[float, float] = (0.9, 0.999)
    use_scheduler: bool = False
    log_every_n_steps: int = 100


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for test-time evaluation, paths, and resampling."""
    preds_dir: str = linear_probe_results_dir / "predictions" # Output directory for predictions
    metrics_dir: str = linear_probe_results_dir / "/metrics"   # Output directory for resampled metrics
    num_resamples: int = 7
    subsample_proportion: float = 0.5
    compute_resampled_metrics_on_test_end: bool = True # Flag to run the full resampling logic

# --- Probing Lightning Module ---

class ProbingLightningModule(LightningModule):
    def __init__(
        self,
        plm_config: ProteinLanguageModelConfig,
        linear_config: LinearModelConfig,
        training_config: TrainingConfig,
        evaluation_config: EvaluationConfig,
        *,
        datamodule,
        seed: int = SEED,
    ):
        """
        Initializes the ProbingLightningModule.

        Args:
            plm_config: Configuration for the Protein Language Model.
            linear_config: Configuration for the Linear Probe.
            training_config: Configuration for training (learning rate, etc.).
            evaluation_config: Configuration for evaluation (prediction saving, resampling).
            datamodule: The LightningDataModule instance, used for task info.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.plm_config = plm_config
        self.linear_config = linear_config
        self.training_config = training_config
        self.eval_config = evaluation_config # Stored
        self.seed = seed
        self.datamodule = datamodule
        self.rng = np.random.default_rng(self.seed)

        self.save_hyperparameters(
            dict(
                plm_config=dataclasses.asdict(plm_config),
                linear_config=dataclasses.asdict(linear_config),
                training_config=dataclasses.asdict(training_config),
                evaluation_config=dataclasses.asdict(evaluation_config),
            )
        )

        self.protein_language_model: ProteinLanguageModel | None = None
        self.linear_probe: LinearProbe | None = None
        self.naive_protein_language_model: ProteinLanguageModel | None = None
        self.untrained_linear_probe: LinearProbe | None = None

        self.test_predictions_storage = []

    def configure_model(self) -> None:
        """Initialize the models, including naive versions for controls."""
        if self.protein_language_model is not None:
            return

        logger.info(f"Configuring models on device: {self.device}")
        torch.manual_seed(self.seed)

        # Trained PLM
        self.protein_language_model = hydra_zen.instantiate(self.plm_config).eval().to(self.device)
        # Naive (un-trained) PLM for control
        self.naive_protein_language_model = hydra_zen.instantiate(self.plm_config)
        reset_to_pytorch_defaults(self.naive_protein_language_model)
        self.naive_protein_language_model.eval().to(self.device)

        # Freeze parameters here just in case
        for param in self.protein_language_model.parameters():
            param.requires_grad = False
        for param in self.naive_protein_language_model.parameters():
            param.requires_grad = False

        hidden_size = self.protein_language_model.model.config.hidden_size

        # Configure linear probe
        self.linear_config.input_dim = hidden_size
        self.linear_config.output_dim = self.datamodule.num_targets
        self.linear_config.purpose = self.datamodule.task_type
        
        # Trained linear probe
        self.linear_probe = hydra_zen.instantiate(self.linear_config).eval().to(self.device)
        
        # Un-trained (naive) linear probe for control
        self.untrained_linear_probe = hydra_zen.instantiate(self.linear_config)
        reset_to_pytorch_defaults(self.untrained_linear_probe)
        self.untrained_linear_probe.eval().to(self.device)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Saves ONLY the linear probe's state_dict and the optimizer state.
        The PLM is excluded and will be re-loaded from pre-trained source on restore.
        """
        
        # 1. Save the linear probe's state_dict under the standard 'state_dict' key
        # We save it here because it is the *only* part of the model we want to save.
        probe_state_dict = self.linear_probe.state_dict()
        checkpoint["state_dict"] = {f"linear_probe.{k}": v for k, v in probe_state_dict.items()}
        
        logger.info("Checkpoint saved contains only the linear probe and optimizer states.")

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Loads the linear probe's state_dict from the checkpoint.
        The PLM is automatically initialized as pre-trained in `configure_model`.
        """
        if "state_dict" in checkpoint:
            # The saved state_dict has keys prefixed with 'linear_probe.'
            state_dict = checkpoint["state_dict"]
            
            # Filter the state_dict to only include the keys for the linear probe
            # We must load the linear probe *after* it's been initialized in `configure_model`
            
            # Initialize models if they haven't been (e.g., if loading a checkpoint before training starts)
            if self.protein_language_model is None:
                self.configure_model()  

            # Create a clean state_dict for the linear_probe (strip the prefix)
            linear_probe_state_dict = {
                k.replace("linear_probe.", ""): v
                for k, v in state_dict.items()
                if k.startswith("linear_probe.")
            }
            
            # Load the state into the linear probe
            if linear_probe_state_dict:
                self.linear_probe.load_state_dict(linear_probe_state_dict, strict=False)
                logger.info("Successfully loaded state for linear_probe.")
            else:
                logger.warning("No state found for linear_probe in checkpoint. Using randomly initialized probe.")

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        """Override to load only the linear probe, ignoring missing keys for the frozen PLMs."""
        # This will call on_load_checkpoint, which handles the probe.
        # We explicitly set strict=False for the full module loading to ignore the PLM keys.
        super().load_state_dict(state_dict, strict=False)


    def get_embeddings(self, sequences: list[str], plm: ProteinLanguageModel, normalize_embeddings:bool=False) -> torch.Tensor:
        """Helper to get embeddings from a PLM, handling task level logic."""
        # This implementation assumes the PLM class handles the layer_to_use from its config
        with torch.no_grad():
            hidden_state, attention_mask, _ = plm(sequences)
            
            # Apply attention mask to zero out padding tokens
            if not plm.apply_attention_mask:
                embeddings = hidden_state * attention_mask.unsqueeze(-1)
            else:
                embeddings = hidden_state
            # 
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            if self.datamodule.task_level == 'protein_level': 
                embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            return embeddings.detach()

    # --- Metric Computation & Logging ---

    def _compute_predictions_for_metrics(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts model outputs to predictions and ensures targets are formatted."""
        logits = outputs["outputs"]
        task_level = self.datamodule.task_level
        task_type = self.datamodule.task_type

        if task_level == 'amino_acid_level':
            # remove BOS/EOS tokens and permute
            start_token, end_token = self.protein_language_model.get_special_token_counts()
            logits = logits[:, start_token : start_token + targets.shape[1], :]
            logits = logits.permute(0,2,1)
            if task_type == 'binary_classification':
                logits = logits.squeeze()
                targets = targets.to(torch.float32)

        # Convert logits to predictions
        if task_type == 'binary_classification':
            predictions = torch.sigmoid(logits)
        elif task_type == 'multiclass_classification':
            # For amino_acid_level, logits are [batch, num_classes, seq_len] after permute
            # Apply softmax along the class dimension (dim=1)
            if task_level == 'amino_acid_level':
                predictions = torch.softmax(logits, dim=1)
            else:
                predictions = torch.softmax(logits, dim=-1)
        elif task_type == 'regression':
            predictions = logits
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        return predictions, targets

    def _compute_online_metrics(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, prefix: str) -> None:
        """Computes and logs a select set of non-elementwise metrics for train/val."""
        predictions, formatted_targets = self._compute_predictions_for_metrics(outputs, targets)
        
        n_classes = self.linear_probe.output_dim
        metrics_dict = get_metrics_dict(self.datamodule.task_level, self.datamodule.task_type, n_classes)

        for metric_name, metric_fn in metrics_dict.items():
            if 'elementwise' in metric_name or 'confusion' in metric_name:
                continue # Skip complex/elementwise metrics for online logging

            try:
                metric_fn = metric_fn.to(self.device)
                
                # Use hard predictions for accuracy/F1, and probabilities for AUROC/AP
                if metric_name in ('auroc', 'average_precision'):
                    metric_input = predictions
                elif 'classification' in self.datamodule.task_type:
                    if self.datamodule.task_type == 'multiclass_classification':
                        # Multiclass: argmax gives the class index
                        metric_input = torch.argmax(predictions, dim=1)
                    else:
                        # Binary: expects probabilities for AUROC/AP, and 0/1 for others
                        # Use predictions (0-1) for torchmetrics which handles thresholding/rounding
                        metric_input = predictions
                else:
                    metric_input = predictions

                # Ensure targets are long for classification
                metric_targets = formatted_targets.long() if 'classification' in self.datamodule.task_type else formatted_targets
                
                result = metric_fn(metric_input, metric_targets)
                
                self.log(f"{prefix}/{metric_name}", result, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            except Exception as e:
                # logger.warning(f"Could not compute/log {metric_name} for {prefix}: {e}")
                pass

    # --- Training/Validation Steps---

    def _shared_step(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Logic for a single step, shared between training and validation.
        Returns the computed loss AND the model outputs.
        """
        inputs, targets = batch["sequence"], batch["targets"]
        
        # 1. Forward pass
        outputs = self.forward(inputs)

        # 2. Format targets for loss (if necessary)
        if self.datamodule.task_type == 'multiclass_classification':
            targets = targets.long()
            if targets.ndim > 1 and targets.size(-1) == 1: 
                targets = targets.squeeze(-1)

        # 3. Compute loss

        loss = self.compute_loss(outputs, targets)
        
        # Return both loss and outputs
        return loss, outputs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # 1. Perform forward pass and calculate loss
        loss, outputs = self._shared_step(batch)
        
        # 2. Log loss metrics
        self.log_metrics(loss, prefix="train")
        
        # 3. Compute and log non-loss metrics using the *already calculated* outputs
        # Note: We still use torch.no_grad() here because _shared_step might have been run 
        # with gradients enabled if it's the training step's first call.
        with torch.no_grad():
            self._compute_online_metrics(outputs, batch["targets"], prefix="train")
            
        return loss["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # 1. Perform forward pass and calculate loss
        loss, outputs = self._shared_step(batch)
        
        # 2. Log loss metrics
        self.log_metrics(loss, prefix="val")
        
        # 3. Compute and log non-loss metrics
        with torch.no_grad():
            self._compute_online_metrics(outputs, batch["targets"], prefix="val")
            
        return loss["loss"]
    
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            hidden_state, attention_mask, attentions = self.protein_language_model(
                x # pass a whole batch of sequences
            )
            if not self.protein_language_model.apply_attention_mask:
                embeddings = hidden_state * attention_mask.unsqueeze(-1)
            else:
                embeddings = hidden_state

            if self.datamodule.task_level == 'protein_level': 
                embeddings = embeddings.sum(
                    dim=1
                ) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pass # amino_acid_level uses token embeddings

        outputs = self.linear_probe(embeddings)
        return {
            "hidden_state": hidden_state,
            "attention_mask": attention_mask,
            "outputs": outputs,
            "attentions": attentions,
        }

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.datamodule.task_level == 'amino_acid_level':
            start_token, end_token = self.protein_language_model.get_special_token_counts()
            logits = outputs["outputs"][:, start_token : start_token + targets.shape[1], :]
            logits = logits.permute(0,2,1)
            if self.datamodule.task_type == 'binary_classification':
                logits = logits.squeeze()
                targets = targets.to(torch.float32) 
                ignore_mask = (targets != -100.0).float()
        else:
            logits = outputs["outputs"]

        elementwise_loss = self.linear_probe.loss_fn(logits, targets)
        if (self.datamodule.task_level == 'amino_acid_level') and (self.datamodule.task_type == 'binary_classification'):
            elementwise_loss *= ignore_mask
            # Average over non-masked elements only
            average_loss = elementwise_loss.sum() / ignore_mask.sum()
        else:
            average_loss = elementwise_loss.mean()
            
        losses = {"loss": average_loss}
        return losses

    def log_metrics(self, loss: dict[str, torch.Tensor], prefix: str = "") -> None:
        """Log loss metrics."""
        for key, value in loss.items():
            log_key = f"{prefix}/{key}" if prefix else key
            self.log(log_key, value, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.linear_probe.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            eps=self.training_config.adam_epsilon,
            betas=self.training_config.adam_betas,
        )
        return optimizer

    def _scrambling_control(self, inputs: list[str], targets: torch.Tensor, datamodule: Any) -> tuple[list[str], torch.Tensor]:
        """Scramble sequences and adjust targets for the scrambling control."""
        scrambled_inputs = scramble_sequences(inputs)
        
        if datamodule.task_level == 'protein_level':
            if datamodule.task_type == 'regression':
                if 'prot_param' in datamodule.dataset_name:
                    # load the prot param df
                    scrambled_targets_dicts = protein_analysis(scrambled_inputs)
                    df = pl.from_dicts(scrambled_targets_dicts)
                    scrambled_targets_np = df.select([pl.col(c).cast(pl.Float32) for c in datamodule.target_cols]).to_numpy()
                    scaled_np = datamodule.scaler.transform(scrambled_targets_np)
                    scrambled_targets = torch.from_numpy(scaled_np).to(self.device)
                else:
                    scrambled_targets_np = self.rng.permutation(targets.cpu().numpy())
                    scrambled_targets = torch.from_numpy(scrambled_targets_np)
            elif 'classification' in datamodule.task_type:
                scrambled_list = [shuffle_tensor(row) for row in targets]
                scrambled_targets = torch.stack(scrambled_list, dim=0)
                if scrambled_targets.ndim > 1 and scrambled_targets.size(-1) == 1:
                    scrambled_targets = scrambled_targets.squeeze(-1)
        elif datamodule.task_level == 'amino_acid_level':
            if datamodule.task_type == 'multiclass_classification':
                scrambled_targets = torch.vstack([shuffle_tensor(row) for row in targets.cpu()])
            elif datamodule.task_type == 'binary_classification':
                scrambled_targets = torch.zeros(targets.shape, dtype=targets.dtype)

        return scrambled_inputs, scrambled_targets.to(self.device)

    def _random_baseline_control(self, input_embeddings: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates random embeddings from a standard normal distribution."""
        random_embeddings = torch.randn(
            input_embeddings.shape, 
            device=input_embeddings.device, 
            dtype=input_embeddings.dtype
        )
        return random_embeddings, targets

    def _mean_control(self, input_embeddings: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replaces all embeddings with the mean embedding of the batch."""
        mean_embeddings = input_embeddings.mean(dim=0).expand(input_embeddings.shape)
        
        if 'classification' in self.datamodule.task_type:
            # Use mode to get the most common target value
            mean_targets = torch.mode(targets.flatten(), dim=0).values.expand(targets.shape)
        elif self.datamodule.task_type == 'regression':
            mean_targets = targets.mean(dim=0).expand(targets.shape)
        
        return mean_embeddings, mean_targets

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Performs the full evaluation with all controls, storing results."""
        # if not self.protein_language_model:
        #     self.configure_model()
            
        plms = {'trained': self.protein_language_model, 'un-trained': self.naive_protein_language_model}
        linear_models = {'trained': self.linear_probe, 'un-trained': self.untrained_linear_probe}

        inputs, targets = batch["sequence"], batch["targets"].to(self.device)

        # only squeeze targets if it won't flatten a batch-of-1 to 1D sequence
        if self.datamodule.task_type == 'multiclass_classification':
            targets = targets.long()
            if targets.dim() > 1 and targets.shape[-1] == 1:
                # Squeeze the feature dimension only, but preserve the batch/sequence dimensions
                targets = targets.squeeze(-1)

        # Scrambled Inputs/Targets (Control)
        scrambled_inputs, scrambled_targets = self._scrambling_control(inputs, targets, self.datamodule)

        for plm_id, plm in plms.items():
            # Get Embeddings
            embeddings = self.get_embeddings(inputs, plm).to(self.device)
            embeddings_from_scrambled = self.get_embeddings(scrambled_inputs, plm).to(self.device)

            # Define Controls (Embeddings and their corresponding Targets)
            mean_embeddings, mean_targets = self._mean_control(embeddings, targets)
            random_embeddings, random_targets = self._random_baseline_control(embeddings, targets)
            normalized_embeddings, normalized_targets = self.get_embeddings(inputs, plm, normalize_embeddings=True).to(self.device), targets

            controls = {
                'original': {'embeddings': embeddings, 'targets': targets},
                'l2_normalized': {'embeddings': normalized_embeddings, 'targets': normalized_targets},
                'mean': {'embeddings': mean_embeddings, 'targets': mean_targets},
                'scrambled': {'embeddings': embeddings_from_scrambled, 'targets': scrambled_targets},
                'random_gaussian': {'embeddings': random_embeddings, 'targets': random_targets},
            }

            for linear_id, linear in linear_models.items():
                for control_type, control_dict in controls.items():
                    embeddings = control_dict['embeddings']
                    control_targets = control_dict['targets']

                    with torch.no_grad():
                        outputs = linear(embeddings).detach().to(control_targets.device)

                    if outputs.ndim == 2 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze(1)
                    
                    # compute_preds_for_metrics expects a dict
                    outputs_dict={
                        "outputs": outputs,
                    }
                    
                    # Compute predictions
                    predictions, _ = self._compute_predictions_for_metrics(outputs_dict, control_targets)

                    # Store results
                    self.test_predictions_storage.append({
                        'dataset': self.datamodule.dataset_name,
                        'model_name': self.protein_language_model.model_shorthand,
                        'layer_num': self.protein_language_model.layer_to_use,
                        'relative_depth': self.protein_language_model.relative_depth,
                        'plm_state': plm_id,
                        'plm_embeddings_normalized': self.protein_language_model.normalize_embeddings,
                        'linear_probe_state': linear_id,
                        'control_type': control_type,
                        'batch_num': batch_idx,
                        'sequences': inputs,
                        'predictions': predictions.to('cpu').numpy(),
                        'targets': control_targets.to('cpu').numpy(),
                        'predictions_shape': predictions.to('cpu').numpy().shape,
                        'targets_shape': control_targets.to('cpu').numpy().shape,
                    })
        

    # --- On Test Epoch End (Prediction Saving and Resampled Metrics) ---

    def on_test_epoch_end(self) -> None:
        """Saves predictions and computes resampled metrics (from prediction_metrics.py)."""
        if not self.test_predictions_storage:
            logger.warning("No test predictions were collected.")
            return

        # 1. Save Predictions to Parquet
        final_df = pl.DataFrame(self.test_predictions_storage)

        # Flatten arrays and ensure shape is stored as a list of integers
        def flatten_array(array):
            return array.flatten().tolist()
        
        output_df = final_df.with_columns(
            pl.col('predictions').map_elements(flatten_array, return_dtype=pl.List(pl.Float32)),
            pl.col('targets').map_elements(flatten_array, return_dtype=pl.List(pl.Float32)),
            pl.col('predictions_shape').map_elements(lambda s: [int(x) for x in s], return_dtype=pl.List(pl.Int64)),
            pl.col('targets_shape').map_elements(lambda s: [int(x) for x in s], return_dtype=pl.List(pl.Int64)),
        )

        preds_file_name = f"{self.datamodule.dataset_name}_{self.protein_language_model.model_shorthand}_layer_{self.protein_language_model.layer_to_use}_emb_norm_{self.plm_config.normalize_embeddings}_{self.seed}_test_predictions.parquet.gz"
        preds_save_dir = Path(self.eval_config.preds_dir) / f"{self.datamodule.dataset_name}"
        preds_save_path = preds_save_dir / preds_file_name
        
        preds_save_dir.mkdir(parents=True, exist_ok=True)
        output_df.write_parquet(preds_save_path)
        logger.info(f"Test predictions saved to {preds_save_path}")

        # 2. Compute Resampled Metrics
        if self.eval_config.compute_resampled_metrics_on_test_end:

            df = final_df
            
            cols_to_narrow = ['dataset', 'model_name', 'layer_num', 'relative_depth', 'plm_state', 'plm_embeddings_normalized', 'linear_probe_state', 'control_type']
            unique_combinations = df.select(cols_to_narrow).unique()
            
            results_list = []
            rng = np.random.default_rng(seed = self.seed)
            task_level = self.datamodule.task_level
            task_type = self.datamodule.task_type
            
            for combo in unique_combinations.iter_rows(named=True):
                filter_conditions = [pl.col(key) == val for key,val in combo.items()]
                sub_df = df.filter(filter_conditions)

                if sub_df.height == 0:
                    continue

                all_predictions_list = []
                all_targets_list = []
                max_len = 0
                
                # Pre-calculate max_len for AA level padding
                if task_level == 'amino_acid_level':
                    for shape_list in sub_df.get_column('targets_shape'):
                         max_len = max(max_len, shape_list[-1])

                # Collect from batches, padding to max_len
                for batch_row in sub_df.iter_rows(named=True):
                    preds = torch.from_numpy(batch_row['predictions'])
                    targets = torch.from_numpy(batch_row['targets'])
                    
                    if task_level == 'amino_acid_level' and preds.ndim == 3:
                        padding_len = max_len - preds.shape[-1]
                        
                        if padding_len > 0:
                            # Pad predictions (fill with 0s)
                            preds = torch.nn.functional.pad(preds, (0, padding_len), 'constant', 0.0)
                            # Pad targets (fill with -100)
                            targets = torch.nn.functional.pad(targets, (0, padding_len), 'constant', -100.0)

                    all_predictions_list.append(preds.to(self.device))
                    all_targets_list.append(targets.to(self.device))


                all_predictions = torch.concat(all_predictions_list)
                all_targets = torch.concat(all_targets_list)
                
                # Resampling and Metric Calculation
                subsample_size = int(all_predictions.shape[0] * self.eval_config.subsample_proportion)
                n_classes = all_predictions.shape[1] if all_predictions.ndim > 1 else None

                metrics_dict = get_metrics_dict(task_level, task_type, n_classes = n_classes)

                # Move all metric objects to that same device
                for metric_name, metric_obj in metrics_dict.items():
                    metrics_dict[metric_name] = metric_obj.to(all_predictions.device)

                combo_metrics = {m: [] for m in metrics_dict.keys()} 
                combo_metrics['fold'] = []

                for time in range(self.eval_config.num_resamples):
                    # Randomly subsample preds, targets
                    inds = torch.from_numpy(rng.choice(all_predictions.shape[0], size = subsample_size, replace=False))
                    preds_subsample = all_predictions[inds]
                    targets_subsample = all_targets[inds]

                    for metric_name, metric in metrics_dict.items():
                        try:
                            if ('classification' in task_type) and ('elementwise' in metric_name):
                                result = compute_metric(metric, preds_subsample, targets_subsample, task_type, elementwise=True)
                            else:
                                result = compute_metric(metric, preds_subsample, targets_subsample, task_type)
                                if (metric_name in ['pearson_corr_coef', 'spearman_corr_coef']):
                                    result = np.mean(result)
                        except Exception as e:
                            print(e)
                            result = None
                            
                        # Store single float or a list of floats
                        combo_metrics[metric_name].append(result)
                        
                    combo_metrics['fold'].append(time)

                final_dict = {k: [v]*self.eval_config.num_resamples for k,v in combo.items()} | combo_metrics 

                results_list.append(pl.DataFrame(final_dict))

            all_metrics_df = pl.concat(results_list, how="diagonal")
            
            metrics_file_name = f"{self.datamodule.dataset_name}_{self.protein_language_model.model_shorthand}_layer_{self.protein_language_model.layer_to_use}_emb_norm_{self.plm_config.normalize_embeddings}_{self.seed}_test_metrics.parquet.gz"
            metrics_save_dir = Path(self.eval_config.metrics_dir) / f"{self.datamodule.dataset_name}"
            metrics_save_path = metrics_save_dir /  metrics_file_name

            metrics_save_dir.mkdir(parents=True, exist_ok=True)
            all_metrics_df.write_parquet(metrics_save_path)

            logger.info(f"Test metrics saved to {metrics_save_path}")