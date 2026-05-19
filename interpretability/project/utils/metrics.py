"""Evaluation metrics for linear probing and KNN experiments.

This module provides metric computation functions for:
- Classification tasks (accuracy, F1, AUROC, precision, recall, etc.)
- Regression tasks (MSE, MAE, Pearson/Spearman correlation, R²)
- Multi-label tasks (coverage error, label ranking metrics)
- Both protein-level and amino-acid-level predictions
"""
import torch
from torchmetrics import (
    CosineSimilarity,
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
    AUROC,
    ConfusionMatrix,
    F1Score,
    MatthewsCorrCoef,
    Accuracy,
    AveragePrecision,
    Precision,
    Recall,
    Specificity,
)
from typing import Any, Union

# For KNNs
from functools import partial
from sklearn.metrics import (
    roc_auc_score, # auroc equivalent
    f1_score, 
    matthews_corrcoef,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score, 
    coverage_error, 
    label_ranking_loss,
    label_ranking_average_precision_score,
    jaccard_score, 
    )

def get_metrics_dict(task_level: str, task_type: str, n_classes: int | None = None):
    """Get appropriate metrics dictionary based on task level and type."""
    if task_level == 'protein_level':
        if 'classification' in task_type:
            task = task_type.split('_')[0]
            if task == 'binary':
                # For binary, num_classes is usually not needed/omitted
                nc_params = {}
            elif n_classes is None:
                # If n_classes is None and not binary, this is where the error comes from
                raise ValueError("n_classes must be provided for non-binary classification tasks at protein level.")
            else:
                nc_params = {'num_classes': n_classes}
                
            metrics_dict = {
                'confusion': ConfusionMatrix(task=task, normalize='true', **nc_params),
                'matthews_corrcoef': MatthewsCorrCoef(task=task, **nc_params),
                'f1_score': F1Score(task=task, **nc_params),
                'f1_score_elementwise': F1Score(task=task, **nc_params),
                'accuracy': Accuracy(task=task, **nc_params),
                'precision': Precision(task=task, **nc_params),
                'recall': Recall(task=task, **nc_params),
                'specificity': Specificity(task=task, **nc_params),
                'auroc': AUROC(task=task, **nc_params),
            }

            if task != 'binary':
                metrics_dict = metrics_dict | {'matthews_corrcoef_elementwise': MatthewsCorrCoef(task=task, **nc_params), 
                                                'accuracy_elementwise': Accuracy(task=task, **nc_params),}


        elif 'regression' in task_type:
            metrics_dict = {
                'cosine_similarity': CosineSimilarity(reduction='mean'),
                'cosine_similarity_elementwise': CosineSimilarity(reduction='none'),
                'mae': MeanAbsoluteError(),
                # 'mae_elementwise': MeanAbsoluteError(num_outputs = n_classes),
                'mse': MeanSquaredError(),
                'mse_elementwise': MeanSquaredError(num_outputs = n_classes),
                'pearson_corr_coef': PearsonCorrCoef(num_outputs = n_classes),
                'pearson_corr_coef_elementwise': PearsonCorrCoef(num_outputs = n_classes),
                'r2_score': R2Score(),
                'r2_score_elementwise': R2Score(num_outputs = n_classes, multioutput='raw_values'),
                'spearman_corr_coef': SpearmanCorrCoef(num_outputs = n_classes),
                'spearman_corr_coef_elementwise': SpearmanCorrCoef(num_outputs = n_classes),
            }
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    elif task_level == 'amino_acid_level':
        # Assume classification
        task = task_type.split('_')[0]
        metrics_dict = {
            'confusion': ConfusionMatrix(task=task, num_classes = n_classes, ignore_index=-100, normalize='true'),
            'matthews_corrcoef': MatthewsCorrCoef(task=task, num_classes = n_classes, ignore_index=-100),
            'f1_score': F1Score(task=task, num_classes = n_classes, ignore_index=-100),
            'accuracy': Accuracy(task=task, num_classes = n_classes, ignore_index=-100),
            'precision': Precision(task=task, num_classes = n_classes, ignore_index=-100),
            # 'precision_elementwise': Precision(task=task),
            'average_precision': AveragePrecision(task=task, num_classes = n_classes, ignore_index=-100),
            # 'average_precision_elementwise': AveragePrecision(task=task),
            'recall': Recall(task=task, num_classes = n_classes, ignore_index=-100),
            # 'recall_elementwise': Recall(task=task),
            'specificity': Specificity(task=task, num_classes = n_classes, ignore_index=-100),
            # 'specificity_elementwise': Specificity(task=task),
            'auroc': AUROC(task=task, num_classes = n_classes, ignore_index=-100),
            # 'auroc_elementwise': AUROC(task=task),
        }
    
    else:
        raise ValueError(f"Unknown task_level: {task_level}")
    
    return metrics_dict


def compute_metric(metric_fn: Any, preds: torch.Tensor, targets: torch.Tensor, task_type: str, elementwise: bool = False) -> Union[float, list[float]]:
    """Compute metric (from metrics.py)."""
    if targets.ndim > 1 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1) 
    
    if 'classification' in task_type:
        # Convert targets to long for classification metrics
        targets = targets.to(torch.long)
        
        # Handle shape mismatch for amino_acid_level multiclass: preds [batch, num_classes, seq_len] vs targets [batch, seq_len]
        if preds.ndim == 3 and targets.ndim == 2 and preds.shape[0] == targets.shape[0] and preds.shape[2] == targets.shape[1]:
            # Get metric class name to determine if it needs probabilities or class predictions
            # Metrics like AUROC and AveragePrecision need probabilities, others need class predictions
            metric_class_name = metric_fn.__class__.__name__.lower()
            needs_probabilities = 'auroc' in metric_class_name or 'averageprecision' in metric_class_name
            
            if elementwise or needs_probabilities:
                # For elementwise or probability-based metrics, reshape to [batch*seq_len, num_classes] and [batch*seq_len]
                batch_size, num_classes, seq_len = preds.shape
                preds = preds.permute(0, 2, 1).reshape(batch_size * seq_len, num_classes)
                targets = targets.reshape(batch_size * seq_len)
            else:
                # Convert probabilities to class predictions using argmax along class dimension
                preds = torch.argmax(preds, dim=1)  # [batch, num_classes, seq_len] -> [batch, seq_len]
                # Flatten for metrics that expect flattened inputs
                if metric_fn.full_state_update:
                    preds = preds.flatten()
                    targets = targets.flatten()
        elif metric_fn.full_state_update and targets.ndim == preds.ndim and targets.ndim > 1:
             # Flatten everything for correct input to torchmetrics with ignore_index
             preds = preds.flatten()
             targets = targets.flatten()
    elif preds.shape != targets.shape:
        print(f"Warning: Final shape mismatch: preds {preds.shape} vs targets {targets.shape}")
             
    if elementwise:
        result = []
        # Go through columns (classes or outputs)
        for i in range(preds.shape[1]):
            # Targets are typically un-squeezed for elementwise
            result.append(metric_fn(preds[:, i], targets[:, i]))
        result = torch.tensor(result)

    else:
        result = metric_fn(preds, targets)
        
    try:
        # Return single float
        return float(result.item())
    except Exception:
        # Return list of floats for matrices/arrays (like ConfusionMatrix or elementwise results)
        if hasattr(result, 'numpy'):
            return [float(x) for x in result.cpu().numpy().flatten()]
        else:
            return [float(x) for x in result.cpu().flatten()]


macro_f1 = partial(f1_score, average='macro', zero_division=0)
macro_auroc = partial(roc_auc_score, average='macro')
macro_precision = partial(precision_score, average='macro', zero_division=0)
macro_recall = partial(recall_score, average='macro', zero_division=0)
macro_jaccard = partial(jaccard_score, average='macro', zero_division=0)

def calculate_labelwise_macro(y_true, y_pred, metric_fn, **kwargs):
    """
    Calculates a binary metric for each label independently and returns the macro-average.
    
    Args:
        y_true: Ground truth (N, C) matrix or (N,) vector.
        y_pred: Predictions (N, C) matrix or (N,) vector.
        metric_fn: The sklearn-style metric function to apply to each column.
        **kwargs: Additional arguments to pass to the metric_fn.
    """
    import numpy as np
    # Handle 1D (standard binary) case
    if y_true.ndim == 1:
        return metric_fn(y_true, y_pred, **kwargs)
        
    # Multi-label case (N x C matrices)
    num_labels = y_true.shape[1]
    scores = []
    
    for i in range(num_labels):
        # Slice the i-th label
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        
        # Calculate score for this specific label
        score = metric_fn(y_true_col, y_pred_col, **kwargs)
        
        # Append only valid numerical results (ignores NaN from ill-defined metrics)
        if not np.isnan(score):
            scores.append(score)
            
    return np.mean(scores) if scores else 0.0

knn_metrics = {
    'coverage_error': 
        lambda y_true, y_pred_proba: coverage_error(y_true, y_pred_proba),
    
    'label_ranking_average_precision_score': 
        lambda y_true, y_pred_proba: label_ranking_average_precision_score(y_true, y_pred_proba),
    
    'label_ranking_loss':
        lambda y_true, y_pred_proba: label_ranking_loss(y_true, y_pred_proba),

    'f1_score': 
        lambda y_true, y_pred_binary: macro_f1(y_true, y_pred_binary),
    
    'accuracy': 
        lambda y_true, y_pred_binary: calculate_labelwise_macro(y_true, y_pred_binary, accuracy_score),

    'precision': 
        lambda y_true, y_pred_binary: macro_precision(y_true, y_pred_binary),
    
    'recall': 
        lambda y_true, y_pred_binary: macro_recall(y_true, y_pred_binary),
    
    'jaccard_score': 
        lambda y_true, y_pred_binary: macro_jaccard(y_true, y_pred_binary),

    'matthews_corrcoef': 
        lambda y_true, y_pred_binary: calculate_labelwise_macro(y_true, y_pred_binary, matthews_corrcoef),
}