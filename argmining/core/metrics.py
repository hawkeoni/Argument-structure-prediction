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
        self.total += predictions.numel()
        binarized_predictions = (predictions >= self.threshold).float()
        assert len(binarized_predictions.shape) == 1
        assert len(gold_labels.shape) == 1
        self.correct += (binarized_predictions == gold_labels).sum().item()