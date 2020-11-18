from typing import Dict, Any

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import F1Measure
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from argmining.core import *
from argmining.core.metrics import ThresholdAccuracy


@Model.register("BasicEntailmentModel")
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
        self.positive_label = vocab.get_token_to_index_vocabulary("labels")["Evidence"]
        self.f1 = F1Measure(positive_label=self.positive_label)

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

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        logits = output_dict["logits"]
        classes = logits.argmax(dim=1).detach().cpu().numpy()
        idx_to_class = self.vocab.get_index_to_token_vocabulary("labels")
        output_dict["result"] = [idx_to_class[cls] for cls in classes]
        return output_dict


@Model.register("StanceModel")
class TopicSentenceClassifier(Model):

    default_predictor = "claim_stance_predictor"

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder = None,
                 encoder: Seq2VecEncoder = None,
                 dropout: float = 0.3,
                 threshold: float = 0.5):
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
        self.clf = nn.Linear(self.encoder.get_output_dim(), 1)
        self.loss = nn.BCELoss()
        self.threshold = threshold
        self.accuracy = ThresholdAccuracy(threshold)

    def forward(self,
                tokens: Dict[str, Dict[str, torch.LongTensor]],
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedded = self.embedder(tokens)
        embedded = self.dropout(embedded)
        encoded_cls = self.encoder(embedded, mask)
        logits = self.clf(encoded_cls).squeeze(1)
        # logits - batch_size
        probs = torch.sigmoid(logits)
        output_dict = {"probs": probs, "logits": logits}
        if labels is not None:
            labels = labels.view(-1)
            loss = self.loss(probs, labels)
            output_dict["loss"] = loss
            self.accuracy(probs, labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.accuracy.get_metric(reset)
        return metrics

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        new_dict = {}
        probs = output_dict["probs"] # number from [0; 1]
        binarized = (probs > self.threshold).detach().cpu().tolist()
        binarized = ["PRO" if p else "CON" for p in binarized]
        new_dict["class"] = binarized
        new_dict["score"] = -1 + probs * 2
        return new_dict