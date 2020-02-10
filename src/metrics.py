from typing import Dict

import torch
from allennlp.training.metrics.metric import Metric


class F1ForSentenceClassification(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._thr = threshold

    def reset(self):
        self._tp = 0
        self._fp = 0
        self._fn = 0

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        precision = self._tp / (self._tp + self._fp) if (self._tp + self._fp > 0) else 0.
        recall = self._tp / (self._tp + self._fn) if (self._tp + self._fn > 0) else 0.
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall > 0) else 0
        return {"precision": precision, "recall": recall, "f1": fscore}

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.LongTensor):
        """
        :param predictions: - float tensor of shape [batch_size]
        :param gold_labels: - long tensor of shape [batch_size]
        """
        predictions = (predictions > self._thr).long()
        self._tp += (predictions * gold_labels).sum().item()
        self._fp += (predictions * (1 - gold_labels)).sum().item()
        self._fn += ((1 - predictions) * gold_labels).sum().item()
