import os
from typing import List, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.models.model import Model
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import Embedding
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.common import Params

from src.readers import ClaimsReader
from src.models import SimpleClassifier


def evaluate(model: Model, test_dataset: List[Instance]) -> Dict[str, Any]:
    model.eval()
    model.get_metrics(reset=True)
    batch_size = 32
    for i in range(len(test_dataset) // batch_size + 1):
        batch = test_dataset[i * batch_size: (i + 1) * batch_size]
        if not batch:
            break
        model.forward_on_instances(batch)
    return model.get_metrics()


def evaluate_and_write(model: Model, test_dataset: List[Instance], filename: str) -> Dict[str, Any]:
    model.eval()
    try:
        model.get_metrics(reset=True)
    except:
        pass
    batch_size = 32
    f = open(filename, "w", encoding="utf8")
    for i in range(len(test_dataset) // batch_size + 1):
        batch = test_dataset[i * batch_size: (i + 1) * batch_size]
        if not batch:
            break
        probs = model.forward_on_instances(batch)
        for sample, p in zip(batch, probs):
            text = " ".join([token.text for token in sample["tokens"].tokens])
            label = sample["labels"].array[0]
            string_name = ""
            prob = p["probs"]
            if label == 1 and prob > 0.5:
                string_name = "true_positive"
            if label == 0 and prob > 0.5:
                string_name = "false_positive"
            if label == 1 and prob < 0.5:
                string_name = "false_negative"
            if label == 0 and prob < 0.5:
                string_name = "true_negative"
            f.write(f"{text}\t{string_name}\t{label}\t{p['probs']}\n")
    f.close()
    return model.get_metrics(True)


def save_results(model: Model, vocab: Vocabulary, res: Dict[str, float], foldername: str):
    try:
        os.mkdir(foldername)
    except BaseException as e:
        print(f"Folder {foldername} already existed, rewriting internals.")
    torch.save(model.state_dict(), os.path.join(foldername, "model.pt"))
    vocab.save_to_files(os.path.join(foldername, "vocab"))
    with open(os.path.join(foldername, "results.txt"), "w") as res_file:
        for k, v in res.items():
            res_file.write(f"{k} = {v}\n")
    print(f"Results saved in {foldername}")


def train_and_evaluate(train_test_claims: pd.DataFrame,
                       validation_claims: pd.DataFrame,
                       articles_path: str,
                       test_topic: str,
                       foldername: str) -> Dict[str, float]:
    train_df = train_test_claims[train_test_claims.Topic != test_topic]
    test_df = train_test_claims[train_test_claims.Topic == test_topic]
    val_df = validation_claims
    print(len(train_df), len(test_df), len(val_df))
    reader = ClaimsReader()
    train_dataset = list(reader._read(train_df, articles_path))
    test_dataset = list(reader._read(test_df, articles_path))
    val_dataset = list(reader._read(val_df, articles_path))
    emb_filename = "embeddings/glove.42B.300d.txt"
    vocab = Vocabulary.from_instances(train_dataset + test_dataset + val_dataset,
                                      pretrained_files={"tokens": emb_filename},
                                      only_include_pretrained_words=True)
    char_emb_dim = 30
    char_hid_dim = 25
    hid_dim = 256

    charemb = Embedding(vocab.get_vocab_size("chars"), char_emb_dim)
    charrnn = PytorchSeq2VecWrapper(nn.LSTM(char_emb_dim, char_hid_dim, batch_first=True, bidirectional=True))
    charrnn = TokenCharactersEncoder(charemb, charrnn)
    emb_params = Params({"pretrained_file": emb_filename, "embedding_dim": 300, "trainable": False})
    embedding = Embedding.from_params(vocab, emb_params)
    embedder = BasicTextFieldEmbedder({"tokens": embedding, "chars": charrnn})
    encoder = PytorchSeq2VecWrapper(nn.LSTM(embedder.get_output_dim(), hid_dim, bidirectional=True, batch_first=True))
    model = SimpleClassifier(vocab, embedder, encoder)
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    devicenum = 0
    iterator = BucketIterator([("tokens", "num_tokens")], biggest_batch_first=True)
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      patience=3,
                      num_epochs=20,
                      cuda_device=devicenum,
                      validation_metric="+f1")
    print(trainer.train())
    res = evaluate(model, test_dataset)
    print(res)
    save_results(model, vocab, res, foldername)
    return res
