from typing import Dict, Any

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Auc, F1Measure


@Model.register("topic_sentence_clf")
class TopicSentenceClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 dropout: float = 0.3):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(self.encoder.get_output_dim(), 2)
        self.loss = nn.CrossEntropyLoss()
        self.f1 = F1Measure(positive_label=1)
        self.auc = Auc()

    def forward(self,
                tokens: Dict[str, Dict[str, torch.LongTensor]],
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        embedded = self.embedder(tokens)
        embedded = self.dropout(embedded)
        mask = get_text_field_mask(tokens)
        # encoded - [batch_size, seq_len, hid_dim]
        encoded = self.encoder(embedded, mask)
        # encoded_cls - [batch_size, hid_dim]
        encoded_cls = encoded[:, 0]
        logits = self.clf(encoded_cls)
        probs = torch.softmax(logits, dim=1)
        output_dict = {"probs": probs, "logits": logits}
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
        metrics.update(self.f1.get_metric(reset))
        return metrics
