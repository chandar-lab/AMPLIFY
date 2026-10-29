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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier

from torch.utils.data import DataLoader, Dataset

from apperitif.evaluation.Evaluator import Evaluator, EvaluatorConfig
from apperitif.models.Encoder import SequenceEncoder
from apperitif.models.regressor import Regressor, MLP
from apperitif.utils.deterministic import set_seed
from apperitif.utils.embedding_utils import get_embeddings_and_labels
from apperitif.utils.optimization import get_linear_scheduler_with_warmup, EarlyStopping
from apperitif.utils.dtype_utils import get_torch_dtype
from apperitif.utils.evaluator_utils import EvaluatorResult, create_standardized_result


@dataclass
class SequenceClassificationEvaluatorConfig(EvaluatorConfig):
    head_type: str = "base"  # Discriminator field for Hydra
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "roc_auc"])
    decision_metric: Optional[str] = None
    num_labels: Optional[int] = None
    reduction_method: str = "mean"
    target_column: str = "labels"
    precomputing_batch_size: int = 128
    autocast_precision: str = "fp16"  # Changed to string for Hydra compatibility


@dataclass
class KNNSequenceClassificationEvaluatorConfig(SequenceClassificationEvaluatorConfig):
    head_type: str = "knn"  # Discriminator field for Hydra
    k: Optional[int | list[int]] = 1


@dataclass
class LinearProbingSequenceClassificationEvaluatorConfig(SequenceClassificationEvaluatorConfig):
    head_type: str = "linear"  # Discriminator field for Hydra
    C: float | list[float] = 1.0  # Inverse of regularization strength for LogisticRegression
    max_iter: int = 1000  # Maximum number of iterations


@dataclass
class MLPSequenceClassificationEvaluatorConfig(SequenceClassificationEvaluatorConfig):
    head_type: str = "mlp"  # Discriminator field for Hydra
    hidden_size: int | list[int] = 512
    learning_rate: float | list[float] = 1e-3
    num_epochs: int | list[int] = 2
    weight_decay: float | list[float] = 0
    batch_size: int | list[int] = 32
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    bayes: bool = False
    n_trials: int = 50
    finetuning: bool = False
    precomputed: bool = True  # Moved to config


class SequenceClassificationEvaluator(Evaluator):
    """
    Base evaluator for sequence classification tasks.
    """

    def __init__(
        self,
        config: SequenceClassificationEvaluatorConfig,
        datasets: Dict[str, Dataset],
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
        assert all(metric in ["accuracy", "f1", "mcc", "roc_auc"] for metric in self.metrics), "Invalid metric(s)."
        assert len(self.metrics) > 0, "At least one metric must be provided."

        if config.decision_metric is None:
            self.decision_metric = self.metrics[0]
        else:
            assert config.decision_metric in self.metrics, f"Invalid decision metric: {config.decision_metric}"
            self.decision_metric = config.decision_metric

        self.num_labels = config.num_labels
        assert self.num_labels > 1, "Number of labels must be greater than 1."
        self.reduction_method = config.reduction_method
        self.target_column = config.target_column
        self.precomputing_batch_size = config.precomputing_batch_size
        self.autocast_precision = get_torch_dtype(config.autocast_precision)
        self.head_type = config.head_type  # For standardized output

    @property
    def name(cls):
        return "sequence_classification"

    def _compute_metric(self, labels: np.ndarray, preds: np.ndarray, metric: str) -> float:
        """
        Compute the value of the given metric.

        Args:
            labels: True labels.
            preds: Predicted probabilities or logits.
            metric: Metric to compute.

        Returns:
            Value of the metric.
        """
        if metric == "accuracy":
            return accuracy_score(labels, np.argmax(preds, axis=1))
        elif metric == "f1":
            return f1_score(labels, np.argmax(preds, axis=1), average="weighted")
        elif metric == "mcc":
            return matthews_corrcoef(labels, np.argmax(preds, axis=1))
        elif metric == "roc_auc":
            if self.num_labels == 2:
                return roc_auc_score(labels, preds[:, 1])
            else:
                return roc_auc_score(labels, preds, multi_class="ovo", labels=np.arange(self.num_labels))
        else:
            raise ValueError(f"Invalid metric: {metric}")

    def __call__(self, model: SequenceEncoder, **kwargs) -> EvaluatorResult:
        """
        Evaluate the model on a sequence classification task.

        Args:
            model: Encoder model to evaluate.

        Returns:
            Dictionary containing the evaluation metrics on the test set.
        """
        raise NotImplementedError


class KNNSequenceClassificationEvaluator(SequenceClassificationEvaluator):
    def __init__(
        self,
        config: KNNSequenceClassificationEvaluatorConfig,
        datasets: Dict[str, Dataset],
        collate_fn: Callable,
    ):
        super().__init__(config, datasets, collate_fn)
        self.k = [config.k] if isinstance(config.k, int) else config.k or [1]
        assert "validation" in self.datasets or len(self.k) == 1, "Validation set is required for k selection."

    @property
    def name(cls):
        return "knn_classification"

    def __call__(
        self,
        model: SequenceEncoder,
        model_kwargs: Dict[str, Any] = {},
        return_all_scores: bool = False,
        do_test: bool = True,
        verbose: bool | str = "reduced",
    ) -> EvaluatorResult:
        dataloaders = {
            name: DataLoader(
                ds,
                batch_size=self.precomputing_batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
            )
            for name, ds in self.datasets.items()
        }

        train_embeddings, train_labels = get_embeddings_and_labels(
            model=model,
            dataloader=dataloaders["train"],
            target_column=self.target_column,
            reduction_method=self.reduction_method,
            model_kwargs=model_kwargs,
            autocast_precision=self.autocast_precision,
        )

        # Only get test embeddings if we need them
        if do_test and "test" in dataloaders:
            test_embeddings, test_labels = get_embeddings_and_labels(
                model=model,
                dataloader=dataloaders["test"],
                target_column=self.target_column,
                reduction_method=self.reduction_method,
                model_kwargs=model_kwargs,
                autocast_precision=self.autocast_precision,
            )

        all_scores = {}
        # Grid search over k values using validation set
        val_embeddings, val_labels = get_embeddings_and_labels(
            model=model,
            dataloader=dataloaders["validation"],
            target_column=self.target_column,
            reduction_method=self.reduction_method,
            model_kwargs=model_kwargs,
            autocast_precision=self.autocast_precision,
        )

        for k in self.k:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train_embeddings, train_labels)

            # Validation score for k selection
            val_preds = knn.predict_proba(val_embeddings)
            val_score = self._compute_metric(val_labels, val_preds, self.decision_metric)

            # Test scores (if requested)
            if do_test and "test" in dataloaders:
                test_preds = knn.predict_proba(test_embeddings)
                test_scores = {metric: self._compute_metric(test_labels, test_preds, metric) for metric in self.metrics}
            else:
                test_scores = {metric: -1 for metric in self.metrics}

            # Store results for this k value
            all_scores[(k,)] = {
                "validation_score": val_score,
                "validation_std": 0.0,  # No std for single run
                "test_scores": test_scores,
                "test_std": {metric: 0.0 for metric in self.metrics},
            }

        # Select the best k based on validation score
        best_k = max(all_scores, key=lambda x: all_scores[x]["validation_score"])
        best_val_score = all_scores[best_k]["validation_score"]
        best_test_scores = all_scores[best_k]["test_scores"]

        self.logger.info(f"Best k: {best_k}")

        # Prepare all_results for detailed analysis (if requested)
        all_results_dict = all_scores if return_all_scores else None

        return create_standardized_result(
            metrics=best_test_scores,
            evaluator_type=self.head_type,
            best_hyperparameters=(best_k,),
            std_metrics={metric: 0.0 for metric in self.metrics},  # No std for single run
            all_results=all_results_dict,
            validation_score=best_val_score,
            validation_std=0.0,  # No std for single run
            model_name=getattr(model, "name_or_path", ""),
            dataset_name=getattr(self, "dataset_name", ""),
        )


class LinearProbingSequenceClassificationEvaluator(SequenceClassificationEvaluator):
    def __init__(
        self,
        config: LinearProbingSequenceClassificationEvaluatorConfig,
        datasets: Dict[str, Dataset],
        collate_fn: Callable,
    ):
        super().__init__(config, datasets, collate_fn)
        self.C = config.C
        self.max_iter = config.max_iter

        # Ensure C is a list for hyperparameter search
        if isinstance(self.C, float):
            self.C = [self.C]

    @property
    def name(cls):
        return "logistic_regression"

    def __call__(
        self,
        model: SequenceEncoder,
        model_kwargs: Dict[str, Any] = {},
        return_all_scores: bool = False,
        do_test: bool = True,
        verbose: bool | str = "reduced",
    ) -> EvaluatorResult:
        dataloaders = {
            name: DataLoader(
                ds,
                batch_size=self.precomputing_batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
            )
            for name, ds in self.datasets.items()
        }

        train_embeddings, train_labels = get_embeddings_and_labels(
            model=model,
            dataloader=dataloaders["train"],
            target_column=self.target_column,
            reduction_method=self.reduction_method,
            model_kwargs=model_kwargs,
            autocast_precision=self.autocast_precision,
        )
        validation_embeddings, validation_labels = get_embeddings_and_labels(
            model=model,
            dataloader=dataloaders["validation"],
            target_column=self.target_column,
            reduction_method=self.reduction_method,
            model_kwargs=model_kwargs,
            autocast_precision=self.autocast_precision,
        )

        # Only get test embeddings if we need them
        if do_test and "test" in dataloaders:
            test_embeddings, test_labels = get_embeddings_and_labels(
                model=model,
                dataloader=dataloaders["test"],
                target_column=self.target_column,
                reduction_method=self.reduction_method,
                model_kwargs=model_kwargs,
                autocast_precision=self.autocast_precision,
            )

        # Hyperparameter search over C values (unified for single and multiple values)
        all_scores = {}
        for c_val in self.C:
            clf = LogisticRegression(C=c_val, max_iter=self.max_iter)
            clf.fit(train_embeddings, train_labels)
            val_preds = clf.predict_proba(validation_embeddings)

            # Compute validation score using the decision metric
            val_score = self._compute_metric(validation_labels, val_preds, self.decision_metric)

            # Compute test scores (if requested)
            if do_test and "test" in dataloaders:
                test_preds = clf.predict_proba(test_embeddings)
                test_scores = {metric: self._compute_metric(test_labels, test_preds, metric) for metric in self.metrics}
            else:
                test_scores = {metric: -1 for metric in self.metrics}

            all_scores[c_val] = (val_score, test_scores)

        # Find the best C based on validation score
        best_c = max(all_scores, key=lambda x: all_scores[x][0])
        best_val_score, best_test_scores = all_scores[best_c]

        self.logger.info(f"Best C: {best_c}")
        self.logger.info(f"Best validation score: {best_val_score}")
        self.logger.info(f"Best test scores: {best_test_scores}")

        # Prepare all_results for detailed analysis (if requested)
        all_results_dict = None
        if return_all_scores:
            all_results_dict = {}
            for c_val, (val_score, test_scores) in all_scores.items():
                all_results_dict[c_val] = {
                    "validation_score": val_score,
                    "test_scores": test_scores,
                }

        return create_standardized_result(
            metrics=best_test_scores,
            evaluator_type=self.head_type,
            best_hyperparameters=best_c,
            all_results=all_results_dict,
            validation_score=best_val_score,
            model_name=getattr(model, "name_or_path", ""),
            dataset_name=getattr(self, "dataset_name", ""),
        )


class MLPSequenceClassificationEvaluator(SequenceClassificationEvaluator):
    def __init__(
        self,
        config: MLPSequenceClassificationEvaluatorConfig,
        datasets: Dict[str, Dataset],
        collate_fn: Callable,
    ):
        super().__init__(config, datasets, collate_fn)
        self.hidden_size = config.hidden_size
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        self.finetuning = config.finetuning  # Whether to fine-tune the trunk or not
        self.precomputed = config.precomputed  # Whether to use precomputed embeddings
        assert not self.precomputed if self.finetuning else True, "Cannot finetune with precomputed embeddings."

        if isinstance(self.hidden_size, int):
            self.hidden_size = [self.hidden_size]
        if isinstance(self.learning_rate, float):
            self.learning_rate = [self.learning_rate]
        if isinstance(self.num_epochs, int):
            self.num_epochs = [self.num_epochs]
        if isinstance(self.weight_decay, float):
            self.weight_decay = [self.weight_decay]
        if isinstance(self.batch_size, int):
            self.batch_size = [self.batch_size]

        self.seeds = config.seeds
        self.bayes = config.bayes
        self.n_trials = config.n_trials
        set_seed(self.seeds[0])

        self.early_stopping = EarlyStopping(patience=16, mode="max", disable=False)

    @property
    def name(cls):
        return "mlp_classification"

    def _evaluate_mlp_precomputed(
        self, mlp: torch.nn.Module, embedding_loader: DataLoader, device: str = "cuda", do_roc_auc: bool = False,
        return_arrays: bool = False,
    ):
        mlp.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in embedding_loader:
                embeddings, batch_labels = batch
                embeddings = embeddings.to(device)
                logits = mlp(embeddings)
                all_logits.append(logits.float().cpu().numpy())
                all_labels.append(batch_labels.long().cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probas = torch.nn.functional.softmax(torch.tensor(all_logits), dim=1).numpy()

        scores = {}
        for metric in self.metrics:
            if metric == "roc_auc" and not do_roc_auc and metric != self.decision_metric:
                scores[metric] = -1
            else:
                scores[metric] = self._compute_metric(all_labels, all_probas, metric)
        # Per-sequence probe predictions (item 1): only when explicitly asked
        # by the caller (final-test path in _train_mlp_precomputed).
        if return_arrays:
            return scores, all_logits, all_labels
        return scores

    def _train_mlp_precomputed(
        self,
        mlp: torch.nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        criterion: Callable,
        num_epochs: int,
        device: str = "cuda",
        do_test: bool = True,
        verbose: bool | str = "reduced",
        accumulation_steps: int = 1,
    ) -> Tuple[float, Dict[str, float]]:
        # Fresh patience for every training run (each HPO trial, seed re-run and grid config).
        self.early_stopping.reset()
        disable = (not verbose) or verbose == "reduced"
        pbar = tqdm.tqdm(
            total=(num_epochs) * len(train_loader) // accumulation_steps, desc="Training MLP", disable=disable
        )
        best_val_score = -float("inf")
        loader_len = len(train_loader)
        should_stop = False
        completed_steps = 0

        with torch.autocast(device_type=device, dtype=torch.float32, enabled=True):
            for n in range(num_epochs):
                mlp.train()
                optimizer.zero_grad()
                for idx, batch in enumerate(train_loader):
                    embeddings, labels = batch
                    embeddings, labels = embeddings.to(device), labels.to(device).long()
                    preds = mlp(embeddings)
                    loss = criterion(preds, labels)
                    loss = loss / accumulation_steps  # Normalize loss for accumulation
                    loss.backward()

                    if (idx + 1) % accumulation_steps == 0 or (idx + 1 == loader_len):
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                        completed_steps += 1
                        pbar.update(1)

                    if ((idx + 1) % (loader_len // 3) == 0) or (idx + 1 == loader_len):
                        val_scores = self._evaluate_mlp_precomputed(
                            mlp, validation_loader, device=device, do_roc_auc=False
                        )
                        val_score = val_scores[self.decision_metric]
                        pbar.set_postfix({"val_score": val_score})
                        if val_score > best_val_score:
                            best_val_score = val_score
                            best_state_dict = {k: v.cpu() for k, v in mlp.state_dict().items()}
                        should_stop = self.early_stopping(val_score)
                        if should_stop:
                            break
                        mlp.train()
                if should_stop:
                    break
            pbar.close()

        if do_test:
            mlp.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
            test_scores, _test_logits, _test_labels = self._evaluate_mlp_precomputed(
                mlp, test_loader, device=device, do_roc_auc=True, return_arrays=True,
            )
            # Per-seq predictions dump (item 1).
            _save_dir = getattr(self, "_test_pred_dir", None)
            if _save_dir:
                try:
                    os.makedirs(_save_dir, exist_ok=True)
                    np.savez_compressed(
                        os.path.join(_save_dir, "test_predictions.npz"),
                        logits=_test_logits, labels=_test_labels,
                    )
                    # Per-seq interpretability sidecar (halt/entropy/cos-sim)
                    # aligned with test set order.  Skipped for non-ponder ckpts.
                    _pss = getattr(self, "_test_persseq_arrays", None)
                    if _pss:
                        np.savez_compressed(
                            os.path.join(_save_dir, "test_interpretability.npz"),
                            **_pss,
                        )
                except Exception as _e:
                    self.logger.warning(f"[per-seq save failed] {_e}")
        else:
            test_scores = {metric: -1 for metric in self.metrics}
        return best_val_score, test_scores, completed_steps

    def _evaluate_classifier(
        self, classifier: Regressor, dataloader: DataLoader, device: str = "cuda", do_roc_auc: bool = False
    ) -> Dict[str, float]:
        classifier.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop(self.target_column).to(device).long().view(-1)
                logits = classifier(batch, frozen_backbone=True)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probas = torch.nn.functional.softmax(torch.tensor(all_logits), dim=1).numpy()

        scores = {}
        for metric in self.metrics:
            if metric == "roc_auc" and not do_roc_auc and metric != self.decision_metric:
                scores[metric] = -1
            else:
                scores[metric] = self._compute_metric(all_labels, all_probas, metric)
        return scores

    def _train_classifier(
        self,
        classifier: Regressor,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        criterion: Callable,
        num_epochs: int,
        finetuning: bool = False,
        device: str = "cuda",
        do_test: bool = True,
        verbose: bool | str = "reduced",
        accumulation_steps: int = 1,
    ) -> Tuple[float, Dict[str, float]]:
        # Fresh patience for every training run (each HPO trial, seed re-run and grid config).
        self.early_stopping.reset()
        disable_pbar = (not verbose) or verbose == "reduced"
        pbar = tqdm.tqdm(
            total=(num_epochs) * len(train_loader) // accumulation_steps,
            desc="Training Classifier",
            disable=disable_pbar,
        )
        best_val_score = -float("inf")
        loader_len = len(train_loader)
        should_stop = False
        completed_steps = 0

        for n in range(num_epochs):
            frozen_backbone = not finetuning or n == 0
            classifier.train()
            for idx, batch in enumerate(train_loader):
                labels = batch.pop(self.target_column).to(device).long()
                preds = classifier(batch, frozen_backbone=frozen_backbone)
                loss = criterion(preds, labels.reshape(-1)) / accumulation_steps  # Normalize loss for accumulation
                loss.backward()

                if (idx + 1) % accumulation_steps == 0 or (idx + 1 == loader_len):
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                    completed_steps += 1
                    pbar.update(1)

                if ((idx + 1) % (loader_len // 3) == 0) or (idx + 1 == loader_len):
                    val_scores = self._evaluate_classifier(
                        classifier, validation_loader, device=device, do_roc_auc=False
                    )
                    val_score = val_scores[self.decision_metric]
                    pbar.set_postfix({"val_score": val_score})
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_state_dict = {k: v.cpu() for k, v in classifier.state_dict().items()}
                    should_stop = self.early_stopping(val_score)
                    if should_stop:
                        break
            if should_stop:
                break

        pbar.close()

        if do_test:
            classifier.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
            test_scores = self._evaluate_classifier(classifier, test_loader, device=device, do_roc_auc=True)
        else:
            test_scores = {metric: -1 for metric in self.metrics}
        return best_val_score, test_scores, completed_steps

    def _run_bayesian_search(
        self, model, train_dataset, validation_dataset, test_dataset,
        train_collate_fn, device, do_test, verbose, accumulation_steps,
        output_folder, task_name, task_description,
    ):
        # Item 1: expose output_folder so _train_mlp_precomputed's final-test
        # path can dump test_predictions.npz there for per-seq analysis.
        self._test_pred_dir = output_folder
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        db_path = os.path.join(output_folder, "optuna_study.db") if output_folder else None
        storage = f"sqlite:///{db_path}" if db_path else None
        study = optuna.create_study(
            direction="maximize", storage=storage,
            study_name="mlp_classification", load_if_exists=True,
        )
        n_remaining = max(0, self.n_trials - len(study.trials))
        self.logger.info(
            f"[bayes] {len(study.trials)} trials already done; running {n_remaining} more (target={self.n_trials})."
        )

        # Convert from OmegaConf ListConfig to plain Python lists so optuna
        # can serialize the choices to SQLite without type errors.
        hidden_sizes   = list(self.hidden_size)  if hasattr(self.hidden_size,  '__iter__') else [self.hidden_size]
        learning_rates = list(self.learning_rate) if hasattr(self.learning_rate, '__iter__') else [self.learning_rate]
        num_epochs_l   = list(self.num_epochs)   if hasattr(self.num_epochs,   '__iter__') else [self.num_epochs]
        weight_decays  = list(self.weight_decay)  if hasattr(self.weight_decay,  '__iter__') else [self.weight_decay]
        batch_sizes    = list(self.batch_size)    if hasattr(self.batch_size,    '__iter__') else [self.batch_size]

        def objective(trial):
            hs = trial.suggest_categorical("hidden_size",  hidden_sizes)
            lr = trial.suggest_float("learning_rate", min(learning_rates), max(learning_rates), log=True)
            ne = trial.suggest_categorical("num_epochs",   num_epochs_l)
            wd = trial.suggest_float("weight_decay",
                                     max(1e-8, min(weight_decays)),
                                     max(max(weight_decays), 1e-8), log=True)
            bs = trial.suggest_categorical("batch_size",   batch_sizes)

            accum = max(1, bs // 8)
            train_loader = DataLoader(train_dataset, batch_size=bs // accum, shuffle=True, collate_fn=train_collate_fn)
            val_loader   = DataLoader(validation_dataset, batch_size=bs // accum, shuffle=False, collate_fn=train_collate_fn)

            set_seed(self.seeds[0])
            mlp = MLP(model.hidden_size, hs, self.num_labels).to(device)
            opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd, eps=1e-4)
            sched = get_linear_scheduler_with_warmup(opt, int(0.1 * ne * len(train_loader)), ne * len(train_loader))
            criterion = torch.nn.CrossEntropyLoss()
            self.early_stopping.reset()
            val_score, _, _ = self._train_mlp_precomputed(
                mlp, train_loader, val_loader, None, opt, sched, criterion, ne,
                device=device, do_test=False, verbose=False, accumulation_steps=accum,
            )
            return val_score

        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining, show_progress_bar=bool(verbose))

        best = study.best_trial.params
        self.logger.info(f"[bayes] Best params: {best}")
        hs, lr, ne, wd, bs = (best["hidden_size"], best["learning_rate"],
                               best["num_epochs"], best["weight_decay"], best["batch_size"])

        accum = max(1, bs // 8)
        train_loader = DataLoader(train_dataset, batch_size=bs // accum, shuffle=True, collate_fn=train_collate_fn)
        val_loader   = DataLoader(validation_dataset, batch_size=bs // accum, shuffle=False, collate_fn=train_collate_fn)
        test_loader  = (DataLoader(test_dataset, batch_size=bs // accum, shuffle=False, collate_fn=train_collate_fn)
                        if do_test and test_dataset is not None else None)

        seed_results = []
        for seed in self.seeds:
            set_seed(seed)
            mlp = MLP(model.hidden_size, hs, self.num_labels).to(device)
            opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd, eps=1e-4)
            sched = get_linear_scheduler_with_warmup(opt, int(0.1 * ne * len(train_loader)), ne * len(train_loader))
            criterion = torch.nn.CrossEntropyLoss()
            self.early_stopping.reset()
            val_score, test_scores, completed_steps = self._train_mlp_precomputed(
                mlp, train_loader, val_loader, test_loader, opt, sched, criterion, ne,
                device=device, do_test=do_test, verbose=verbose, accumulation_steps=accum,
            )
            seed_results.append((val_score, test_scores, completed_steps))

        import numpy as np
        mean_val  = float(np.mean([r[0] for r in seed_results]))
        std_val   = float(np.std([r[0] for r in seed_results]))
        mean_test = {m: float(np.mean([r[1][m] for r in seed_results])) for m in self.metrics}
        std_test  = {"std_" + m: float(np.std([r[1][m] for r in seed_results])) for m in self.metrics}
        best_steps = min(r[2] for r in seed_results)
        best_hparam_key = str((hs, lr, ne, wd, bs))

        self.logger.info(f"[bayes] Best val: {mean_val:.4f} ± {std_val:.4f}")
        self.logger.info(f"[bayes] Best test: {mean_test}")

        all_scores = {best_hparam_key: {
            "validation_score": mean_val, "test_scores": mean_test,
            "validation_std": std_val, "test_std": std_test,
            "min_completed_steps": best_steps,
        }}
        if output_folder:
            self.save_results_to_json(all_scores, output_folder, task_name, task_description)

        std_metrics = {k.replace("std_", ""): v for k, v in std_test.items()}
        _sp = [f"task={task_name or 'unknown'}"]
        for _m, _v in sorted(mean_test.items()):
            _sp.append(f"{_m}={_v:.6f}±{std_metrics.get(_m, 0.0):.6f}")
        _sp.append(f"val={mean_val:.6f}±{std_val:.6f}")
        _sp.append(f"hparams={best_hparam_key}")
        _sp.append("search=bayes")
        print("SCORE | " + " | ".join(_sp), flush=True)
        return create_standardized_result(
            metrics=mean_test, evaluator_type=self.head_type,
            best_hyperparameters=best_hparam_key, std_metrics=std_metrics,
            all_results=all_scores, validation_score=mean_val, validation_std=std_val,
            model_name=getattr(model, "name_or_path", ""),
        )

    def __call__(
        self,
        model: SequenceEncoder,
        model_kwargs: Dict[str, Any] = {},
        device: str = "cuda",
        return_all_scores: bool = False,
        do_test: bool = True,
        verbose: bool | str = "reduced",
        accumulation_steps: int = None,
        output_folder: str = None,
        task_name: Optional[str] = None,
        task_description: Optional[str] = None,
    ) -> EvaluatorResult:
        if self.precomputed:
            precomputing_dataloaders = {
                name: DataLoader(
                    ds,
                    batch_size=self.precomputing_batch_size,
                    collate_fn=self.collate_fn,
                    shuffle=False,
                )
                for name, ds in self.datasets.items()
                if name in ["train", "validation", "test"]
            }

            train_embeddings, train_labels = get_embeddings_and_labels(
                model,
                precomputing_dataloaders["train"],
                self.target_column,
                self.reduction_method,
                model_kwargs,
                self.autocast_precision,
            )
            val_embeddings, val_labels = get_embeddings_and_labels(
                model,
                precomputing_dataloaders["validation"],
                self.target_column,
                self.reduction_method,
                model_kwargs,
                self.autocast_precision,
            )
            if do_test and "test" in precomputing_dataloaders:
                test_embeddings, test_labels = get_embeddings_and_labels(
                    model,
                    precomputing_dataloaders["test"],
                    self.target_column,
                    self.reduction_method,
                    model_kwargs,
                    self.autocast_precision,
                )
                # Item 1 extension: capture per-test-seq ponder halt arrays
                # (populated in model._last_persseq_arrays by pop_halt_stats
                # at the end of get_embeddings_and_labels above).  Consumed
                # in _train_mlp_precomputed's final-test save block.
                self._test_persseq_arrays = dict(getattr(model, "_last_persseq_arrays", {}) or {})

            train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_embeddings), torch.tensor(train_labels))
            validation_dataset = torch.utils.data.TensorDataset(torch.tensor(val_embeddings), torch.tensor(val_labels))
            if do_test and "test" in precomputing_dataloaders:
                test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_embeddings), torch.tensor(test_labels))
            train_collate_fn = None
        else:
            train_dataset, validation_dataset = self.datasets["train"], self.datasets["validation"]
            test_dataset = self.datasets.get("test") if do_test else None
            train_collate_fn = self.collate_fn

        if self.bayes:
            return self._run_bayesian_search(
                model=model,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset if do_test else None,
                train_collate_fn=train_collate_fn,
                device=device, do_test=do_test, verbose=verbose,
                accumulation_steps=accumulation_steps,
                output_folder=output_folder,
                task_name=task_name, task_description=task_description,
            )

        all_scores = {}
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

            print("Complete HP grid:", hyperparameters)
            hyperparameters = set(hyperparameters) - set(hyperparameters_done)

            print("Remaining HP grid:", hyperparameters)
            self.logger.info(
                f"Found {len(hyperparameters_done)} previously computed hyperparameters. Only {len(hyperparameters)} remaining to compute."
            )

            if len(hyperparameters) == 0:
                self.logger.info("All hyperparameters already computed. Finishing job.")
                return None

        global_pbar = tqdm.tqdm(
            total=len(hyperparameters),
            desc="Hyperparameter search",
            disable=(not verbose) or (verbose != "reduced"),
            smoothing=0,
        )
        for hparam in hyperparameters:
            hidden_size, learning_rate, num_epochs, weight_decay, batch_size = hparam
            all_scores[hparam] = []

            if accumulation_steps is None or batch_size % accumulation_steps != 0:
                accumulation_steps = max(1, batch_size // 8)

            print("Hyperparameters:", hparam)
            print("Accumulation steps:", accumulation_steps)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size // accumulation_steps,  # Adjust batch size for accumulation steps
                shuffle=True,
                collate_fn=train_collate_fn,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size // accumulation_steps,
                shuffle=False,
                collate_fn=train_collate_fn,
            )
            if do_test and test_dataset is not None:
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size // accumulation_steps,
                    shuffle=False,
                    collate_fn=train_collate_fn,
                )
            else:
                test_loader = None

            for seed in self.seeds:
                print("Seed:", seed)
                set_seed(seed)
                if self.precomputed:
                    mlp = MLP(model.hidden_size, hidden_size, self.num_labels).to(device)
                    optimizer = torch.optim.AdamW(
                        mlp.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-4
                    )
                elif not self.finetuning:
                    model.eval()
                    classifier = Regressor(
                        trunk=model,
                        hidden_size=hidden_size,
                        output_size=self.num_labels,
                        reduction_method=self.reduction_method,
                        trunk_autocast_precision=self.autocast_precision,
                        head_precision=torch.float32,
                    ).to(device)
                    optimizer = torch.optim.AdamW(
                        classifier.head.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-4
                    )
                else:  # Finetuning
                    model.train()
                    classifier = Regressor(
                        trunk=model,
                        hidden_size=hidden_size,
                        output_size=self.num_labels,
                        reduction_method=self.reduction_method,
                        trunk_autocast_precision=self.autocast_precision,
                        head_precision=torch.float32,
                    ).to(device)
                    optimizer = torch.optim.AdamW(
                        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-4
                    )

                scheduler = get_linear_scheduler_with_warmup(
                    optimizer,
                    num_warmup_steps=int(0.1 * num_epochs * len(train_loader)),
                    num_training_steps=num_epochs * len(train_loader),
                )
                criterion = torch.nn.CrossEntropyLoss()
                self.early_stopping.reset()

                if self.precomputed:
                    val_score, test_scores, completed_steps = self._train_mlp_precomputed(
                        mlp,
                        train_loader,
                        validation_loader,
                        test_loader,
                        optimizer,
                        scheduler,
                        criterion,
                        num_epochs,
                        device,
                        do_test,
                        verbose=verbose,
                        accumulation_steps=accumulation_steps,
                    )
                else:  # Handles both finetuning and head-only training
                    val_score, test_scores, completed_steps = self._train_classifier(
                        classifier,
                        train_loader,
                        validation_loader,
                        test_loader,
                        optimizer,
                        scheduler,
                        criterion,
                        num_epochs,
                        self.finetuning,
                        device,
                        do_test,
                        verbose=verbose,
                        accumulation_steps=accumulation_steps,
                    )
                all_scores[hparam].append((val_score, test_scores, completed_steps))

            mean_val_score = np.mean([s[0] for s in all_scores[hparam]])
            std_val_score = np.std([s[0] for s in all_scores[hparam]])
            mean_test_score = {
                metric: np.mean([s[1][metric] for s in all_scores[hparam] if s[1] != -1]) for metric in self.metrics
            }
            std_test_score = {
                metric: np.std([s[1][metric] for s in all_scores[hparam] if s[1] != -1]) for metric in self.metrics
            }
            completed_steps = min([s[2] for s in all_scores[hparam]])
            all_scores[hparam] = {
                "validation_score": mean_val_score,
                "validation_std": std_val_score,
                "test_scores": mean_test_score,
                "test_std": std_test_score,
                "min_completed_steps": completed_steps,
            }
            global_pbar.update(1)

            # Checkpoint
            self.save_results_to_json(
                all_scores=all_scores,
                output_folder=output_folder,
                task_name=task_name,
                task_description=task_description,
            )

        global_pbar.close()

        best_hyperparameters = max(all_scores, key=lambda x: all_scores[x]["validation_score"])
        best_val_score = all_scores[best_hyperparameters].get("validation_score", 0)
        best_test_scores = all_scores[best_hyperparameters].get("test_scores", {})
        best_std_val_score = all_scores[best_hyperparameters].get("validation_std", 0)
        best_std_test_scores = all_scores[best_hyperparameters].get("test_std", {})
        best_completed_steps = all_scores[best_hyperparameters].get("min_completed_steps", -1)

        self.logger.info(f"Best hyperparameters: {best_hyperparameters}")
        self.logger.info(f"Best validation score: {best_val_score:.4f} +- {best_std_val_score:.4f}")
        self.logger.info(f"Best test scores: {best_test_scores} +- {best_std_test_scores}")
        self.logger.info(f"Best completed steps: {best_completed_steps}")

        # Convert std_test_scores format (remove "std_" prefix)
        std_metrics = {k.replace("std_", ""): v for k, v in best_std_test_scores.items()}
        _sp = [f"task={task_name or 'unknown'}"]
        for _m, _v in sorted(best_test_scores.items()):
            _sp.append(f"{_m}={_v:.6f}±{std_metrics.get(_m, 0.0):.6f}")
        _sp.append(f"val={best_val_score:.6f}±{best_std_val_score:.6f}")
        _sp.append(f"hparams={best_hyperparameters}")
        _sp.append("search=grid")
        print("SCORE | " + " | ".join(_sp), flush=True)
        return create_standardized_result(
            metrics=best_test_scores,
            evaluator_type=self.head_type,
            best_hyperparameters=best_hyperparameters,
            std_metrics=std_metrics,
            all_results=all_scores,
            validation_score=best_val_score,
            validation_std=best_std_val_score,
            model_name=getattr(model, "name_or_path", ""),
            dataset_name=getattr(self, "dataset_name", ""),
        )
