import os

import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import Embedding
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.common import Params

from src.readers import ClaimsReader
from src.utils import evaluate_and_write, evaluate, save_results
from src.models import SimpleClassifier


def train_lstm(train_test_path: str, dev_path: str, TOPIC: str, model_dir: str):
    reader = ClaimsReader()
    train_test = reader.read(train_test_path)
    dev_dataset = reader.read(dev_path)
    train_dataset = []
    test_dataset = []
    for instance in train_test:
        if instance["metadata"].metadata["topic"] == TOPIC:
            test_dataset.append(instance)
        else:
            train_dataset.append(instance)
    assert len(test_dataset) > 0
    print(len(train_dataset), len(dev_dataset), len(test_dataset))
    emb_filename = "embeddings/glove.42B.300d.txt"
    vocab = Vocabulary.from_instances(train_dataset + test_dataset + dev_dataset,
                                      pretrained_files={"tokens": emb_filename},
                                      only_include_pretrained_words=True)
    char_emb_dim = 50
    char_hid_dim = 50
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
                      validation_dataset=dev_dataset,
                      patience=3,
                      num_epochs=20,
                      cuda_device=devicenum,
                      validation_metric="+f1",
                      serialization_dir=model_dir)
    print(f"Training metrics of {model_dir}")
    print(trainer.train())
    model.eval()
    model.load_state_dict(torch.load(os.path.join(model_dir, "best.th")))
    res = evaluate(model, test_dataset)
    print("Results on test.")
    print(res)
    save_results(model, vocab, res, model_dir)
    wres = evaluate_and_write(model, test_dataset, os.path.join(model_dir, "predictions.txt"))
    assert wres == res


def main():
    train_test = "train_test.txt"
    dev = "val.txt"
    for i, topic in enumerate(open("train_test_topics.txt")):
        if i >= 14:
            topic = topic.strip()
            train_lstm(train_test, dev, topic, f"model_lstm_{i}")


if __name__ == "__main__":
    main()
