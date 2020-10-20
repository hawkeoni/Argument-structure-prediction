from typing import Dict

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import FBetaMeasure
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from src import BertCLSPooler


@Model.register("topic_sentence_model")
class TopicSentenceClassifier(Model):

    default_predictor = "topic_sentence_predictor"

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder = None,
                 encoder: Seq2VecEncoder = None,
                 dropout: float = 0.3):
        super().__init__(vocab)
        self.embedder = embedder
        if embedder is None:
            embedder = BasicTextFieldEmbedder(
                {
                    "tokens": PretrainedTransformerEmbedder("bert-base-multilingual-cased")
                }
            )
            self.embedder = embedder
        self.encoder = encoder or BertCLSPooler(self.embedder.get_output_dim())
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(self.encoder.get_output_dim(), 2)
        self.loss = nn.CrossEntropyLoss()
        self.f1 = FBetaMeasure(average="micro")

    def forward(self,
                tokens: Dict[str, Dict[str, torch.LongTensor]],
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedded = self.embedder(tokens)
        embedded = self.dropout(embedded)
        encoded_cls = self.encoder(embedded, mask)
        logits = self.clf(encoded_cls)
        probs = torch.softmax(logits, dim=1)
        output_dict = {"probs": probs, "logits": logits}
        if labels is not None:
            # labels - [batch_size]
            labels = labels.long()
            loss = self.loss(logits, labels)
            output_dict["loss"] = loss
            self.f1(logits, labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.f1.get_metric(reset)
        return metrics
