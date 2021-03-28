from typing import Dict, Any

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import FBetaMeasure, CategoricalAccuracy, F1Measure
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
        self.loss = nn.L1Loss()
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
        pred = torch.tanh(logits)
        output_dict = {"pred": pred, "logits": logits}
        if labels is not None:
            labels = labels.view(-1)
            # labels - batch_size
            loss = self.loss(pred, labels)
            output_dict["loss"] = loss
            self.accuracy(pred, labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.accuracy.get_metric(reset)
        return metrics

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        new_dict = {}
        pred = output_dict["pred"]  # number from [0; 1]
        pred = pred.detach().cpu().tolist()
        binarized = []
        for p in pred:
            if pred >= 0.34:
                binarized.append("PRO")
            elif pred <= -0.34:
                binarized.append("CON")
            else:
                pred.append("NEUTRAL")
        new_dict["class"] = binarized
        new_dict["score"] = pred
        return new_dict

@Model.register("NLIModel")
class NLIModel(Model):

    default_predictor = "NLIPredictor"

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
        num_classes = self.vocab.get_vocab_size("labels")
        assert num_classes > 0, "Wrong namespace for labels apparently"
        self.clf = nn.Linear(self.encoder.get_output_dim(), num_classes)
        self.accuracy = CategoricalAccuracy()
        self.f1 = FBetaMeasure(average=None, labels=list(range(self.vocab.get_vocab_size("labels"))))

    def forward(self,
                tokens: Dict[str, Dict[str, torch.LongTensor]],
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedded = self.embedder(tokens)
        embedded = self.dropout(embedded)
        encoded_cls = self.encoder(embedded, mask)
        logits = self.clf(encoded_cls)
        # logits - batch_size, num_classes
        output_dict = {"logits": logits}
        if labels is not None:
            # labels - batch_size
            labels = labels.view(-1)
            loss = cross_entropy(logits, labels)
            output_dict["loss"] = loss
            self.accuracy(logits, labels)
            self.f1(logits, labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        acc: float = self.accuracy.get_metric(reset)
        metrics["accuracy"] = acc
        f1 = self.f1.get_metric(reset)
        for name, idx in self.vocab.get_token_to_index_vocabulary("labels").items():
            for metric_name, value in f1.items():
                metrics[name + "_" + metric_name] = value[idx]
        return metrics

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        return output_dict
