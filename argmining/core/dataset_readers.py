import json
from typing import Dict, List, Iterable

import pandas as pd
import numpy as np
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Tokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.fields import TextField, LabelField, MetadataField, ArrayField


@DatasetReader.register("NLIReader")
class NLIReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ):
        super().__init__(lazy)
        assert isinstance(
            tokenizer, PretrainedTransformerTokenizer
        ), "Tokenizer must be a transformer."
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

    def _read(self, filepath: str) -> Iterable[Instance]:
        for line in open(filepath):
            d = json.loads(line)
            if d["gold_label"] == "-":
                continue
            yield self.text_to_instance(d["sentence1"], d["sentence2"], d["gold_label"])

    def text_to_instance(
        self, text: str, hypothesis: str, label: str = None
    ) -> Instance:
        fields = {}
        # to make it look like [cls] sent1 [sep] sent2 [sep]
        tokens1 = self._tokenizer.tokenize(text)
        tokens2 = self._tokenizer.tokenize(hypothesis)[1:]
        for token in tokens2:
            token.type_id = 1
        tokens = tokens1 + tokens2
        fields["tokens"] = TextField(tokens, self._token_indexers)

        if label is not None:
            fields["labels"] = LabelField(label)

        return Instance(fields)


@DatasetReader.register("NLIReaderVectorized")
class NLIReaderVectorized(NLIReader):
    def text_to_instance(
        self, text: str, hypothesis: str, label: str = None
    ) -> Instance:
        fields = {}
        # to make it look like two instances [cls] sent1 [sep] & [cls] sent2 [sep]
        tokens1 = self._tokenizer.tokenize(text)
        tokens2 = self._tokenizer.tokenize(hypothesis)
        fields["tokens1"] = TextField(tokens1, self._token_indexers)
        fields["tokens2"] = TextField(tokens2, self._token_indexers)

        if label is not None:
            fields["labels"] = LabelField(label)

        return Instance(fields)


@DatasetReader.register("NLIReaderSE")
class NLIReaderSE(NLIReader):
    def text_to_instance(
        self, text: str, hypothesis: str, label: str = None
    ) -> Instance:
        fields = {}
        # to make it look like [cls] sent1 [sep] sent2 [sep]
        tokens1 = self._tokenizer.tokenize(text)
        tokens2 = self._tokenizer.tokenize(hypothesis)[1:]
        for token in tokens2:
            token.type_id = 1
        tokens = tokens1 + tokens2
        fields["tokens"] = TextField(tokens, self._token_indexers)
        middle_index = len(tokens1)
        fields["middle"] = ArrayField(np.array([middle_index]))
        if label is not None:
            fields["labels"] = LabelField(label)

        return Instance(fields)
