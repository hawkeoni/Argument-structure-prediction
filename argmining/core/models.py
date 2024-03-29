from typing import Dict, Any

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import (
    FBetaMeasure,
    CategoricalAccuracy
)

from argmining.core.modules import (
    BertCLSPooler,
    SICModel,
    InterpretationModel
)


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
        self.encoder = encoder or BertCLSPooler(self.embedder.get_output_dim())
        self.dropout = nn.Dropout(dropout)
        num_classes = self.vocab.get_vocab_size("labels")
        assert num_classes > 0, "Wrong namespace for labels apparently"
        self.clf = nn.Linear(self.encoder.get_output_dim(), num_classes)
        self.accuracy = CategoricalAccuracy()
        assert num_classes == 2 or num_classes == 3
        labels = list(range(num_classes))
        self.f1 = FBetaMeasure(average=None, labels=labels)

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


@Model.register("NLIModelVectorized")
class NLIModelVectorized(NLIModel):

    default_predictor = "NLIPredictor"

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder = None,
                 encoder: Seq2VecEncoder = None,
                 dropout: float = 0.3,
                 highway: bool = False):
        super().__init__(vocab, embedder, encoder, dropout)
        self.embedder = embedder
        self.encoder = encoder or BertCLSPooler(self.embedder.get_output_dim())
        self.dropout = nn.Dropout(dropout)
        num_classes = self.vocab.get_vocab_size("labels")
        assert num_classes > 0, "Wrong namespace for labels apparently"
        self.highway = highway
        if not highway:
            self.clf = nn.Linear(self.encoder.get_output_dim() * 2, num_classes)
        else:
            output_dim = self.encoder.get_output_dim()
            self.clf = nn.Sequential(nn.Linear(output_dim * 2, output_dim),
                                     nn.Sigmoid(),
                                     nn.Linear(output_dim, num_classes))
        self.accuracy = CategoricalAccuracy()
        self.f1 = FBetaMeasure(average=None, labels=list(range(self.vocab.get_vocab_size("labels"))))

    def forward(self,
                tokens1: Dict[str, Dict[str, torch.LongTensor]],
                tokens2: Dict[str, Dict[str, torch.LongTensor]],
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        mask1 = get_text_field_mask(tokens1)
        embedded1 = self.embedder(tokens1)
        embedded1 = self.dropout(embedded1)

        mask2 = get_text_field_mask(tokens2)
        embedded2 = self.embedder(tokens2)

        encoded1 = self.encoder(embedded1, mask1)
        encoded2 = self.encoder(embedded2, mask2)
        # encodedi - [batch, d_model]
        encoded_cls = torch.cat((encoded1, encoded2), dim=1)

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


@Model.register("NLIModelSE")
class NLIModelSE(Model):

    default_predictor = "NLIPredictor"

    def __init__(self,
                 vocab: Vocabulary,
                 lambd: float,
                 embedder: TextFieldEmbedder = None,
                 dropout: float = 0.3):
        super().__init__(vocab)
        self.lambd = lambd
        self.embedder = embedder
        self.dropout = nn.Dropout(dropout)
        num_classes = self.vocab.get_vocab_size("labels")
        assert num_classes > 0, "Wrong namespace for labels apparently"
        self.sic = SICModel(self.embedder.get_output_dim())
        self.interp = InterpretationModel(self.embedder.get_output_dim())
        self.clf = nn.Linear(self.embedder.get_output_dim(), num_classes)
        self.accuracy = CategoricalAccuracy()
        self.f1 = FBetaMeasure(average=None, labels=list(range(self.vocab.get_vocab_size("labels"))))

    def generate_span_masks(self, middle: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        middle = middle.detach().view(-1).tolist()  # [batch_size, 1]
        # mask - [batch_size, seq_len]
        lengths = mask.detach().sum(dim=1).tolist()
        # lengths - [batch_size]
        maxlen = max(lengths)
        num_spans = (maxlen - 1) * (maxlen - 2) // 2
        # span_masks should be [batch_size, span_num]
        batch_size = mask.shape[0]
        span_masks = [[0 for _ in range(num_spans)] for _ in range(batch_size)]
        """        
        for start_index, end_index in zip(start_indexs, end_indexs):
            if 1 <= start_index <= length.item() - 2 and 1 <= end_index <= length.item() - 2 and (
                start_index > middle_index or end_index < middle_index):
                span_mask.append(0)
            else:
                span_mask.append(1e6)
        """
        for batch_idx in range(batch_size):
            sample_middle: int = middle[batch_idx]
            sample_length: int = lengths[batch_idx]
            column = 0
            for span_start in range(1, maxlen - 1):
                for span_end in range(span_start, maxlen - 1):
                    if 1 <= span_start <= sample_length - 2 and 1 <= span_end <= sample_length - 2 and (
                            span_start > sample_middle or span_end < sample_middle):
                        span_masks[batch_idx][column] = 0
                    else:
                        span_masks[batch_idx][column] = 1e-6
                    column += 1
            assert column == num_spans, f"{column}, {num_spans}, {maxlen}"

        return torch.LongTensor(span_masks).to(mask.device)

    def forward(self,
                tokens: Dict[str, Dict[str, torch.LongTensor]],
                middle: torch.LongTensor,
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedded = self.embedder(tokens)
        embedded = self.dropout(embedded)
        # embedded - [batch, seq_len, d_model]
        hij = self.sic(embedded, mask)
        # hij - [batch, num_spans, d_model]
        span_masks = self.generate_span_masks(middle, mask)
        encoded_cls, alphas = self.interp(hij, span_masks)
        logits = self.clf(encoded_cls)
        # logits - batch_size, num_classes
        output_dict = {"logits": logits, "alphas": alphas}
        if labels is not None:
            # labels - batch_size
            labels = labels.view(-1)
            loss = cross_entropy(logits, labels)
            alpha2 = alphas * alphas
            output_dict["loss"] = loss + self.lambd * alpha2.sum(dim=1).mean()
            self.accuracy(logits, labels)
            self.f1(logits, labels)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        acc: float = self.accuracy.get_metric(reset)
        metrics["accuracy"] = acc
        f1 = self.f1.get_metric(reset)
        token_to_idx = self.vocab.get_token_to_index_vocabulary("labels")
        for name, idx in token_to_idx.items():
            for metric_name, value in f1.items():
                metrics[name + "_" + metric_name] = value[idx]
        return metrics
