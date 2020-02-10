from typing import Dict

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Auc

from src.metrics import F1ForSentenceClassification


class SimpleClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 dropout: float = 0.3,
                 threshold: float = 0.5,
                 feedforward: FeedForward = None):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        if feedforward is not None:
            self.feedforward = feedforward
            clf_dim = self.feedforward.get_output_dim()
        else:
            self.feedforward = None
            clf_dim = self.encoder.get_output_dim()
        self.classifier = nn.Linear(clf_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.auc = Auc()
        self.f1 = F1ForSentenceClassification(threshold)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        embedded = self.embedder(tokens)
        embedded = self.dropout(embedded)
        mask = get_text_field_mask(tokens)
        vector = self.encoder(embedded, mask=mask)
        if self.feedforward is not None:
            vector = self.feedforward(vector)
        logits = self.classifier(vector).squeeze(1)
        probs = torch.sigmoid(logits)
        output_dict = {"probs": probs}
        if labels is not None:
            labels = labels.squeeze(1)
            loss = self.criterion(logits, labels)
            output_dict["loss"] = loss
            self.auc(probs, labels)
            self.f1(probs, labels.long())
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"auc": self.auc.get_metric(reset)}
        metrics.update(self.f1.get_metric(reset))
        return metrics
