import os
import ast
import json
import logging
from typing import List, Dict, Tuple, Any, Callable, Optional
import tqdm
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from apperitif.evaluation.Evaluator import Evaluator, EvaluatorConfig
from apperitif.utils.evaluator_utils import create_standardized_result
from apperitif.utils.embedding_utils import get_token_embeddings_and_contacts
from apperitif.data.TokenEmbeddingsDataset import token_embeddings_collate_fn
from apperitif.models.Encoder import Encoder, StructureEncoder
from apperitif.models.regressor import MLP, Regressor, StructureRegressor
from apperitif.utils.deterministic import set_seed
from apperitif.utils.optimization import get_linear_scheduler_with_warmup, EarlyStopping
from apperitif.utils.metrics import (
    contact_map_loss,
    compute_precisions,
    _contact_scores_from_pred,
    _dense_targets_from_pairs,
    compute_contact_metric_value,
)
from apperitif.utils.dtype_utils import get_torch_dtype


@dataclass
class SequenceStructureEvaluatorConfig(EvaluatorConfig):
    metrics: List[str] = field(
        default_factory=lambda: [
            "local_AUC",
            "local_P@L",
            "local_P@L2",
            "local_P@L5",
            "short_AUC",
            "short_P@L",
            "short_P@L2",
            "short_P@L5",
            "medium_AUC",
            "medium_P@L",
            "medium_P@L2",
            "medium_P@L5",
            "long_AUC",
            "long_P@L",
            "long_P@L2",
            "long_P@L5",
        ]
    )
    decision_metric: Optional[str] = "medium_P@L5"
    reduction_method: str = "mean"
    target_column: str = "labels"
    autocast_precision: str = "fp16"  # Changed to string for Hydra compatibility
    hidden_size: int | list[int] = 512
    learning_rate: float | list[float] = 1e-3
    num_epochs: int | list[int] = 2
    weight_decay: float | list[float] = 0
    batch_size: int | list[int] = 2  # Effective batch size (overall batch size)
    internal_batch_size: int = 2  # Actual batch size per forward pass (must divide batch_size)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    finetuning: bool = False
    l: int = 8
    precomputed: bool = False
    precomputing_batch_size: int = 64
    train_minsep: int = 3  # Remove easy contacts with separation < train_minsep during loss
    loss_pos_weight: float = 30.0
    loss_label_smoothing: float = 0.0
    head_type: str = "mlp"


class SequenceStructureEvaluator(Evaluator):
    def __init__(
        self,
        config: SequenceStructureEvaluatorConfig,
        datasets: Dict[str, Any],
        collate_fn: Callable,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=None, level=logging.INFO)

        self.config = config
        self.datasets = datasets
        self.collate_fn = collate_fn

        # Extract config values
        self.metrics = config.metrics
        self.decision_metric = config.decision_metric or self.metrics[0]
        self.target_column = config.target_column
        self.autocast_precision = get_torch_dtype(config.autocast_precision)
        self.hidden_size = [config.hidden_size] if isinstance(config.hidden_size, int) else config.hidden_size
        self.learning_rate = [config.learning_rate] if isinstance(config.learning_rate, float) else config.learning_rate
        self.num_epochs = [config.num_epochs] if isinstance(config.num_epochs, int) else config.num_epochs
        self.weight_decay = [config.weight_decay] if isinstance(config.weight_decay, float) else config.weight_decay
        self.batch_size = [config.batch_size] if isinstance(config.batch_size, int) else config.batch_size
        self.internal_batch_size = config.internal_batch_size  # Actual batch size per forward pass
        self.seeds = config.seeds
        self.finetuning = config.finetuning
        self.precomputed = config.precomputed  # Add this for consistency with regression evaluator
        self.precomputing_batch_size = config.precomputing_batch_size
        self.early_stopping = EarlyStopping(patience=8, mode="max", disable=False)
        set_seed(self.seeds[0])

    @property
    def name(cls):
        return "structure_mlp"

    def _compute_metric(self, predicted_contact_maps, true_contacts, metric: str) -> float:
        # Interpret full metric name e.g., 'medium_P@L5'
        # Compute per-sequence, then average
        values = []
        for pm, tc in zip(predicted_contact_maps, true_contacts):
            val = compute_contact_metric_value(pm, tc, metric)
            values.append(val)
        return float(np.mean(values)) if values else 0.0

    def _evaluate(
        self,
        model,
        dataloader,
        device: str = "cuda",
        save_predictions: bool = False,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        model.eval()
        metrics = metrics or self.metrics
        metric_lists = {metric: [] for metric in metrics}

        # Parse requested metrics into ranges and stats for efficient computation
        def parse_metric_name(name: str):
            range_value, stat_name = name.split("_", 1)
            if stat_name not in {"AUC", "P@L", "P@L2", "P@L5"}:
                raise ValueError(f"Invalid stat '{stat}' in metric '{name}'")
            return range_value, stat_name

        requested = {}
        for m in metrics:
            range_value, stat = parse_metric_name(m)
            requested.setdefault(range_value, set()).add(stat)

        with torch.no_grad():
            for batch in dataloader:
                true_labels = batch[self.target_column]
                pred_maps = model(
                    batch, frozen_backbone=True, use_precomputed_embeddings=self.precomputed, device_type=device
                )

                for pm, tl in zip(pred_maps, true_labels):
                    # Efficient path: compute scores/targets once per sequence
                    scores = _contact_scores_from_pred(pm)
                    L = scores.shape[0]
                    targets = _dense_targets_from_pairs(tl, L=L, device=scores.device)

                    # For each requested range, compute precisions once and fill stats
                    for rng, stats in requested.items():
                        bounds = {
                            "local": (3, 6),
                            "short": (6, 12),
                            "medium": (12, 24),
                            "long": (24, None),
                        }[rng]
                        res = compute_precisions(scores, targets, minsep=bounds[0], maxsep=bounds[1])
                        for stat in stats:
                            metric_lists[f"{rng}_{stat}"].append(float(res[stat].item()))
        if save_predictions:
            # Save the last predictions for debugging or analysis
            last_predictions = {"predicted_contact_maps": pred_maps, "true_labels": true_labels}
            torch.save(last_predictions, "last_predictions.pt")
        scores = {metric: float(np.mean(values)) if values else 0.0 for metric, values in metric_lists.items()}
        return scores

    def _train(
        self,
        model,
        train_loader,
        val_loader,
        eval_loaders: Dict[str, Any],
        optimizer,
        scheduler,
        criterion,
        num_epochs,
        device: str = "cuda",
        do_test: bool = True,
        verbose: bool | str = "reduced",
        accumulation_steps: int = 1,
    ):
        disable = not verbose
        pbar = tqdm.tqdm(
            total=(num_epochs) * len(train_loader) // accumulation_steps, desc="Training Structure MLP", disable=disable
        )
        best_val_score = -1
        should_stop = False
        all_losses = []
        completed_steps = 0
        for n in range(num_epochs):
            model.train()
            frozen_backbone = not self.finetuning or n == 0
            avg_loss = 0.0
            loss_count = 0
            optimizer.zero_grad()
            for idx, batch in enumerate(train_loader):
                true_labels = batch[self.target_column]
                pred_maps = model(
                    batch,
                    frozen_backbone=frozen_backbone,
                    use_precomputed_embeddings=self.precomputed,
                    device_type=device,
                )
                loss = criterion(pred_maps, true_labels) / accumulation_steps
                loss.backward()

                if (idx + 1) % accumulation_steps == 0 or (idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                    completed_steps += 1
                    pbar.update(1)

                avg_loss += loss.item()
                all_losses.append(loss.item())
                loss_count += 1
                if ((idx + 1) % (len(train_loader) // 3) == 0) or (idx + 1 == len(train_loader)):
                    # Compute validation metrics on the train set
                    # train_scores = self._evaluate(
                    #     model, train_loader, device=device, metrics=[self.decision_metric, "medium_AUC"]
                    # )
                    # Validation: compute only the decision metric for efficiency
                    val_scores = self._evaluate(
                        model, val_loader, device=device, metrics=[self.decision_metric, "medium_AUC"]
                    )
                    val_score = val_scores[self.decision_metric]
                    val_auc = val_scores["medium_AUC"]
                    pbar.set_postfix(
                        {
                            "val_score": val_score,
                            "val_auc_medium": val_auc,
                            "train_loss": avg_loss / loss_count if loss_count > 0 else 0.0,
                            # "train_" + self.decision_metric: train_scores[self.decision_metric],
                            # "train_medium_AUC": train_scores["medium_AUC"],
                        }
                    )
                    avg_loss = 0.0
                    loss_count = 0
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                    should_stop = self.early_stopping(val_score)
                    if should_stop:
                        break
                    model.train()
            if should_stop:
                break
        pbar.close()
        if do_test and eval_loaders:
            model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
            # Evaluate on all eval splits (test, casp16, etc.)
            all_eval_scores = {}
            for split_name, loader in eval_loaders.items():
                split_scores = self._evaluate(
                    model,
                    loader,
                    device=device,
                    save_predictions=(split_name == "test"),  # Only save predictions for main test split
                    metrics=self.metrics,
                )
                all_eval_scores[split_name] = split_scores
        else:
            all_eval_scores = {}
        return best_val_score, all_eval_scores, all_losses, completed_steps

    def __call__(
        self,
        model: StructureEncoder,
        model_kwargs: Dict[str, Any] = {},
        device: str = "cuda",
        return_all_scores: bool = False,
        do_test: bool = True,
        verbose: bool | str = "reduced",
        output_folder: Optional[str] = None,
        task_name: Optional[str] = None,
        task_description: Optional[str] = None,
    ) -> Any:
        self.logger.info("Starting evaluation of the structure regressor.")
        datasets = self.datasets
        all_scores: Dict[Any, Any] = {}

        # Precompute embeddings once if needed (following regression evaluator pattern)
        if self.precomputed:
            self.logger.info("Precomputing embeddings for all datasets...")

            # Identify all eval splits (anything not train/validation)
            eval_split_names = [name for name in datasets.keys() if name not in ["train", "validation"]]

            # Check for cached embeddings
            cache_loaded = False
            if output_folder is not None:
                train_path = os.path.join(output_folder, "train_embeddings.pt")
                val_path = os.path.join(output_folder, "val_embeddings.pt")

                # Check if required files exist (train + validation + all eval splits)
                files_exist = os.path.exists(train_path) and os.path.exists(val_path)
                if do_test:
                    for split_name in eval_split_names:
                        split_path = os.path.join(output_folder, f"{split_name}_embeddings.pt")
                        files_exist = files_exist and os.path.exists(split_path)

                if files_exist:
                    try:
                        self.logger.info(f"Loading cached embeddings from {output_folder}")
                        train_data = torch.load(train_path)
                        val_data = torch.load(val_path)
                        eval_data = {}
                        if do_test:
                            for split_name in eval_split_names:
                                split_path = os.path.join(output_folder, f"{split_name}_embeddings.pt")
                                eval_data[split_name] = torch.load(split_path)
                        cache_loaded = True
                    except Exception as e:
                        self.logger.warning(f"Failed to load cached embeddings: {e}. Recomputing...")

            if not cache_loaded:
                compute_bs = self.precomputing_batch_size
                # Build dataloaders for train, validation, and all eval splits
                splits_to_compute = ["train", "validation"]
                if do_test:
                    splits_to_compute.extend(eval_split_names)

                dataloaders = {
                    name: torch.utils.data.DataLoader(
                        ds,
                        batch_size=compute_bs,
                        collate_fn=self.collate_fn,
                        shuffle=False,
                    )
                    for name, ds in datasets.items()
                    if name in splits_to_compute
                }

                train_data = get_token_embeddings_and_contacts(
                    model,
                    dataloaders["train"],
                    target_column=self.target_column,
                    device=device,
                    autocast_precision=self.autocast_precision,
                )
                val_data = get_token_embeddings_and_contacts(
                    model,
                    dataloaders["validation"],
                    target_column=self.target_column,
                    device=device,
                    autocast_precision=self.autocast_precision,
                )

                # Compute embeddings for all eval splits
                eval_data = {}
                if do_test:
                    for split_name in eval_split_names:
                        eval_data[split_name] = get_token_embeddings_and_contacts(
                            model,
                            dataloaders[split_name],
                            target_column=self.target_column,
                            device=device,
                            autocast_precision=self.autocast_precision,
                        )

                if output_folder is not None:
                    self.logger.info(f"Saving precomputed embeddings to {output_folder}")
                    os.makedirs(output_folder, exist_ok=True)
                    torch.save(train_data, os.path.join(output_folder, "train_embeddings.pt"))
                    torch.save(val_data, os.path.join(output_folder, "val_embeddings.pt"))
                    for split_name, split_data in eval_data.items():
                        torch.save(split_data, os.path.join(output_folder, f"{split_name}_embeddings.pt"))

            # Build simple in-memory datasets as lists of dicts
            train_dataset = [{"embeddings": d["embeddings"], self.target_column: d["labels"]} for d in train_data]
            val_dataset = [{"embeddings": d["embeddings"], self.target_column: d["labels"]} for d in val_data]

            def structure_embeddings_collate_fn(batch_list):
                """Collate function for precomputed structure embeddings."""
                return {
                    "embeddings": [item["embeddings"] for item in batch_list],
                    self.target_column: [item[self.target_column] for item in batch_list],
                }

            # Use the precomputed datasets
            datasets = {
                "train": train_dataset,
                "validation": val_dataset,
            }
            # Add all eval split datasets
            for split_name, split_data in eval_data.items():
                datasets[split_name] = [
                    {"embeddings": d["embeddings"], self.target_column: d["labels"]} for d in split_data
                ]

            collate_fn = structure_embeddings_collate_fn
            self.logger.info("Embeddings precomputed successfully.")
        else:
            # Use the original datasets and collate_fn
            collate_fn = self.collate_fn
            self.logger.info("Using original datasets without precomputing.")

        hyperparameters = [
            (hs, lr, ne, wd, bs)
            for bs in self.batch_size
            for hs in self.hidden_size
            for lr in self.learning_rate
            for ne in self.num_epochs
            for wd in self.weight_decay
        ]

        # Load previously computed hyperparameters if available
        if output_folder is not None and os.path.exists(os.path.join(output_folder, "results.json")):
            file_path = os.path.join(output_folder, "results.json")
            with open(file_path, "rt") as f:
                results_dict = json.load(f)
            all_scores = results_dict["all_results"]
            hyperparameters_done = [ast.literal_eval(s) for s in list(results_dict["all_results"].keys())]

            hyperparameters = set(hyperparameters) - set(hyperparameters_done)
            self.logger.info(
                f"Found {len(hyperparameters_done)} previously computed hyperparameters. Only {len(hyperparameters)} remaining to compute."
            )

        print(hyperparameters)
        for hparam in hyperparameters:
            hidden_size, learning_rate, num_epochs, weight_decay, batch_size = hparam

            # Compute accumulation_steps from effective batch_size and internal_batch_size
            # batch_size is the effective (overall) batch size
            # internal_batch_size is the per-forward-pass batch size
            assert (
                batch_size % self.internal_batch_size == 0
            ), f"batch_size ({batch_size}) must be divisible by internal_batch_size ({self.internal_batch_size})"
            accumulation_steps = batch_size // self.internal_batch_size

            print("Hyperparameters:", hparam)
            print(
                f"Effective batch size: {batch_size}, Internal batch size: {self.internal_batch_size}, Accumulation steps: {accumulation_steps}"
            )

            # Accumulate results over seeds for this hyperparameter set
            all_scores[hparam] = []

            for seed in self.seeds:
                set_seed(seed)

                # Build dataloaders from the (potentially precomputed) datasets
                # Use internal_batch_size for the actual DataLoader batch size
                train_loader = torch.utils.data.DataLoader(
                    datasets["train"],
                    batch_size=self.internal_batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                )
                val_loader = torch.utils.data.DataLoader(
                    datasets["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
                )

                # Build loaders for all eval splits (everything except train/validation)
                eval_loaders = {}
                if do_test:
                    eval_split_names = [name for name in datasets.keys() if name not in ["train", "validation"]]
                    for split_name in eval_split_names:
                        eval_loaders[split_name] = torch.utils.data.DataLoader(
                            datasets[split_name], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
                        )

                structure_regressor = StructureRegressor(
                    trunk=model,
                    hidden_size=hidden_size,
                    output_size=2,
                    head_precision=self.autocast_precision,
                    trunk_autocast_precision=self.autocast_precision,
                ).to(device)

                # Sanity check: ensure head produces two-class logits per pair
                if hasattr(structure_regressor.head, "fc2"):
                    assert (
                        getattr(structure_regressor.head.fc2, "out_features", None) == 2
                    ), "Structure MLP head must output 2 logits (neg,pos)."

                print("Finetuning", self.finetuning)
                if self.finetuning:
                    optimizer = torch.optim.AdamW(
                        structure_regressor.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-4
                    )
                else:
                    optimizer = torch.optim.AdamW(
                        structure_regressor.head.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-4
                    )

                # Calculate total optimization steps accounting for gradient accumulation
                steps_per_epoch = len(train_loader) // accumulation_steps
                if len(train_loader) % accumulation_steps != 0:
                    steps_per_epoch += 1
                total_steps = num_epochs * steps_per_epoch

                scheduler = get_linear_scheduler_with_warmup(
                    optimizer,
                    num_warmup_steps=int(0.1 * total_steps),
                    num_training_steps=total_steps,
                )
                criterion = lambda pred, true: contact_map_loss(
                    pred,
                    true,
                    minsep=self.config.train_minsep,
                    pos_weight=self.config.loss_pos_weight,
                    label_smoothing=self.config.loss_label_smoothing,
                )
                self.early_stopping.reset()

                val_score, eval_scores, losses, completed_steps = self._train(
                    structure_regressor,
                    train_loader,
                    val_loader,
                    eval_loaders,
                    optimizer,
                    scheduler,
                    criterion,
                    num_epochs,
                    device,
                    do_test,
                    verbose=verbose,
                    accumulation_steps=accumulation_steps,
                )
                all_scores[hparam].append((val_score, eval_scores, completed_steps))

            # Aggregate results over seeds for this hparam
            mean_val = np.mean([s[0] for s in all_scores[hparam]]) if all_scores[hparam] else 0.0
            std_val = np.std([s[0] for s in all_scores[hparam]]) if all_scores[hparam] else 0.0
            min_completed_steps = int(min([s[2] for s in all_scores[hparam]])) if all_scores[hparam] else -1

            # Aggregate scores for each eval split
            # s[1] is now a dict of {split_name: {metric: value}}
            eval_split_names = set()
            for s in all_scores[hparam]:
                eval_split_names.update(s[1].keys())

            aggregated_scores = {}
            for split_name in eval_split_names:
                mean_scores = {
                    metric: float(
                        np.mean(
                            [
                                s[1][split_name][metric]
                                for s in all_scores[hparam]
                                if split_name in s[1] and s[1][split_name].get(metric, -1) != -1
                            ]
                        )
                    )
                    for metric in self.metrics
                }
                std_scores = {
                    "std_"
                    + metric: float(
                        np.std(
                            [
                                s[1][split_name][metric]
                                for s in all_scores[hparam]
                                if split_name in s[1] and s[1][split_name].get(metric, -1) != -1
                            ]
                        )
                    )
                    for metric in self.metrics
                }
                aggregated_scores[f"{split_name}_scores"] = mean_scores
                aggregated_scores[f"{split_name}_std"] = std_scores

            all_scores[hparam] = {
                "validation_score": mean_val,
                "validation_std": std_val,
                "min_completed_steps": min_completed_steps,
                **aggregated_scores,
            }

            # Checkpoint results to disk
            if output_folder is not None:
                self.save_results_to_json(
                    all_scores=all_scores,
                    output_folder=output_folder,
                    task_name=task_name or "sequence_structure",
                    task_description=task_description or "Structure evaluation",
                )

        best_hyperparameters = max(all_scores, key=lambda x: all_scores[x]["validation_score"]) if all_scores else None
        best_val_score = all_scores[best_hyperparameters].get("validation_score", 0) if best_hyperparameters else 0
        best_std_val_score = all_scores[best_hyperparameters].get("validation_std", 0) if best_hyperparameters else 0

        # Extract scores for all eval splits from best hyperparameters
        best_eval_scores = {}
        best_std_eval_scores = {}
        if best_hyperparameters:
            for key, value in all_scores[best_hyperparameters].items():
                if key.endswith("_scores") and key != "validation_score":
                    best_eval_scores[key] = value
                elif key.endswith("_std") and key != "validation_std":
                    best_std_eval_scores[key] = value

        # For backward compatibility, use test_scores as the primary metrics if available
        best_test_scores = best_eval_scores.get("test_scores", {})
        best_std_test_scores = best_std_eval_scores.get("test_std", {})

        # Collect all metrics including additional eval splits (casp16_scores, etc.)
        all_metrics = dict(best_test_scores)
        all_std_metrics = dict(best_std_test_scores)

        # Add all eval split scores (casp16_scores, etc.)
        for key, value in best_eval_scores.items():
            if key != "test_scores":  # test_scores already in all_metrics
                all_metrics[key] = value
        for key, value in best_std_eval_scores.items():
            if key != "test_std":  # test_std already in all_std_metrics
                all_std_metrics[key] = value

        # Create standardized result with all metrics
        result = create_standardized_result(
            metrics=all_metrics,
            evaluator_type="structure",
            best_hyperparameters=best_hyperparameters,
            std_metrics=all_std_metrics,
            all_results=all_scores if return_all_scores else None,
            validation_score=best_val_score,
            validation_std=best_std_val_score,
            model_name=getattr(model, "name_or_path", ""),
            dataset_name=getattr(self, "dataset_name", ""),
        )

        return result
