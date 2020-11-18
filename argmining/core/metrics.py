from typing import Dict

import torch

from allennlp.training.metrics import Metric


@Metric.Register("threshold_accuracy")
class ThresholdAccuracy:

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.total = 0
        self.correct = 0

    def reset(self):
        self.total = 0
        self.correct = 0

    def get_metrics(self, reset: bool) -> Dict[str, float]:
        accuracy = self.correct / (self.total + 1e-12)
        if reset:
            self.reset()
        return {"accuracy": accuracy}

    def __call__(
        self, 
        predictions: torch.Tensor, 
        gold_labels: torch.Tensor, 
        mask: Optional[torch.BoolTensor] = None):
    predictions = predictions > self.threshold
    self.total += len(predictions)
    self.correct += (predictions * gold_labels).sum()