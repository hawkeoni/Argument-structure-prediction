from typing import Dict, List, Iterable

import pandas as pd
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Tokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.fields import TextField, LabelField, MetadataField


@DatasetReader.register("IBMReader")
class ClaimsReader(DatasetReader):

    default_model_name = "bert-base-multilingual-cased"

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False):
        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(self.default_model_name)
        cls = self.tokenizer.tokenizer.cls_token
        cls_idx = self.tokenizer.tokenizer.cls_token_id
        sep = self.tokenizer.tokenizer.sep_token
        sep_idx = self.tokenizer.tokenizer.sep_token_id
        self.cls_token = Token(text=cls, text_id=cls_idx)
        self.sep_token = Token(text=sep, text_id=sep_idx)
        self.token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(self.default_model_name)
        }

    def _read(self, filepath: str) -> Iterable[Instance]:
        dfs = pd.read_csv(filepath, chunksize=100)
        for df in dfs:
            for rownum, row in df.iterrow():
                topic = row["topic"]
                sentence = row["candidate"]
                label = row.get("label")
                yield self.text_to_instance(sentence, topic, label)

    def text_to_instance(
            self,
            sentence: str,
            topic: str = None,
            label: int = None
    ) -> Instance:
        fields = {}
        sentence_tokens = self.tokenizer.tokenize(sentence)
        sentence_tokens[0] = self.sep_token
        topic_tokens = [self.cls_token, self.sep_token]
        if topic:
            topic_tokens = self.tokenizer.tokenize(topic)
        tokens = topic_tokens + sentence_tokens
        tokens = tokens[:512]
        tokens[-1] = self.sep_token
        fields["tokens"] = TextField(tokens, token_indexers=self.token_indexers)
        if label is not None:
            fields["labels"] = LabelField(label, skip_indexing=True)
        fields["metadata"] = MetadataField({"topic": topic, "sentence": sentence})
        return Instance(fields)
