from typing import Dict

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Auc, F1Measure


class SimpleClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 dropout: float = 0.4,
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
        self.classifier = nn.Linear(clf_dim, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.auc = Auc()
        self.f1 = F1Measure(positive_label=1)
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
        # vector - [batch_size, hidden_dim]
        logits = self.classifier(vector) # [batch_size, 2]
        probs = torch.softmax(logits, dim=1)[:, 1]
        output_dict = {"probs": probs}
        if labels is not None:
            # labels - [batch_size]
            labels = labels.long()
            loss = self.criterion(logits, labels)
            output_dict["loss"] = loss
            self.auc(probs, labels)
            self.f1(logits, labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"auc": self.auc.get_metric(reset)}
        precision, recall, fscore = self.f1.get_metric(reset)
        metrics.update({"precision": precision, "recall": recall, "f1": fscore})
        return metrics
