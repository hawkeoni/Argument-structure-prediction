from typing import Dict, Optional

import torch

from allennlp.training.metrics import Metric


@Metric.register("threshold_accuracy")
class ThresholdAccuracy:

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.total = 0
        self.correct = 0

    def reset(self):
        self.total = 0
        self.correct = 0

    def get_metric(self, reset: bool) -> Dict[str, float]:
        accuracy = self.correct / (self.total + 1e-12)
        if reset:
            self.reset()
        return {"accuracy": accuracy}

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
        ):
        assert len(predictions.shape) == 1
        assert len(gold_labels.shape) == 1
        self.total += predictions.numel()
        pros = (predictions >= 0.34)
        cons = (predictions <= -0.34)
        neut = (-0.34 < predictions) * (predictions < 0.34)
        assert (pros * cons * neut).sum().item() == 0
        self.correct += (pros * (gold_labels == 1.)).sum().item()
        self.correct += (cons * (gold_labels == -1.)).sum().item()
        self.correct += (neut * (gold_labels == 0.)).sum().item()
