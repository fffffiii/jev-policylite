from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.clip(values / temperature, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray, max_fpr: float) -> float:
    labels = labels.astype(np.int64)
    negatives = int((labels == 0).sum())
    if negatives == 0:
        raise ValueError("校准集必须包含阴性样本")
    reject_all_threshold = float(np.nextafter(probabilities.max(), np.inf))
    candidates = np.unique(np.concatenate(([0.0, reject_all_threshold], probabilities)))
    best_threshold = reject_all_threshold
    best_recall = -1.0
    for threshold in candidates:
        predicted = probabilities >= threshold
        false_positive_rate = float(((predicted == 1) & (labels == 0)).sum()) / negatives
        positives = int((labels == 1).sum())
        recall = (
            float(((predicted == 1) & (labels == 1)).sum()) / positives if positives else 0.0
        )
        if false_positive_rate <= max_fpr and recall > best_recall:
            best_threshold = float(threshold)
            best_recall = recall
    return best_threshold


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float,
    predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    predicted = (probabilities >= threshold).astype(np.int64) if predictions is None else predictions.astype(np.int64)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    result: dict[str, Any] = {
        "count": int(labels.size),
        "threshold": float(threshold) if predictions is None else "per_policy",
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / labels.size if labels.size else 0.0,
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
    }
    if len(np.unique(labels)) == 2:
        result["pr_auc"] = float(average_precision_score(labels, probabilities))
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    return result


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = loss_fn(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))
