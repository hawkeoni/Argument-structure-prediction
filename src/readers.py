from typing import Dict, List, Iterable
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, ArrayField
from nltk.tokenize import sent_tokenize, word_tokenize


class ClaimsReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False):
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True),
                                                 "chars": TokenCharactersIndexer("chars", min_padding_length=3)}

    def text_to_instance(self,
                         tokens: List[Token],
                         label: int = None) -> Instance:
        fields = dict()
        if label is not None:
            fields["labels"] = ArrayField(np.array([label]))
        fields["tokens"] = TextField(tokens, token_indexers=self.token_indexers)
        return Instance(fields)

    def _read(self, claims_df: pd.DataFrame, articles_path: str) -> Iterable[Instance]:
        # TODO: fix _read and read
        positive = 0
        article_ids = claims_df["article Id"].unique()
        for article_id in tqdm(article_ids):
            claims_subdf = claims_df[claims_df["article Id"] == article_id]
            article_text = re.sub("\s+", " ", open(articles_path + f"clean_{article_id}.txt").read())
            article_sentences = sent_tokenize(article_text)
            added_sentences = set()
            for i, claim_row in claims_subdf.iterrows():
                claim_text = re.sub("\s+", " ", claim_row["Claim corrected version"].lower())
                for j, sentence in enumerate(article_sentences):
                    if claim_text in sentence.lower() and j not in added_sentences:
                        positive += 1
                        # TODO: fix maxlen
                        sentence_tokens = [Token(token) for token in word_tokenize(sentence)][:120]
                        added_sentences.add(j)
                        yield self.text_to_instance(sentence_tokens, 1)
            for j, sentence in enumerate(article_sentences):
                if j not in added_sentences:
                    sentence_tokens = [Token(token) for token in word_tokenize(sentence)]
                    yield self.text_to_instance(sentence_tokens, 0)
        print(f"Found {positive} positive sentences of {len(claims_df)}")