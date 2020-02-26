from typing import Dict, List, Iterable

from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Tokenizer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField, MetadataField


class ClaimsReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 limit: int = None,
                 lazy: bool = False):
        super().__init__(lazy=lazy)
        self.limit = limit
        self.tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True),
                                                 "chars": TokenCharactersIndexer("chars", min_padding_length=3)}

    def text_to_instance(self,
                         tokens: List[Token],
                         label: int = None,
                         topic: str = None)-> Instance:
        fields = dict()
        if label is not None:
            fields["labels"] = LabelField(label, skip_indexing=True)
        fields["metadata"] = MetadataField({"topic": topic})
        fields["tokens"] = TextField(tokens, token_indexers=self.token_indexers)
        return Instance(fields)

    def _read(self, filepath: str) -> Iterable[Instance]:
        for line in open(filepath, "r", encoding="utf8"):
            topic, sentence, label = line.strip().split("\t")
            tokens = self.tokenizer.tokenize(sentence)#[:120]
            if self.limit is not None:
                tokens = tokens[:self.limit]
                tokens[-1] = Token("[SEP]")


            yield self.text_to_instance(tokens, int(label), topic)
