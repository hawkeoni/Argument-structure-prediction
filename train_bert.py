import os
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import Embedding
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.models import Model
from allennlp.common import Params
from allennlp.training.learning_rate_schedulers import NoamLR

from src.readers import ClaimsReader
from src.utils import evaluate_and_write, evaluate, save_results
from src.models import SimpleClassifier
from src.modules import BertCLSPooler



def train_bert(train_test_path: str, dev_path: str, TOPIC: str, model_dir: str):
    modelname = "bert-base-cased"
    tokenizer = PretrainedTransformerTokenizer(modelname, do_lowercase=False)
    reader = ClaimsReader(token_indexers={"tokens":PretrainedTransformerIndexer(modelname, do_lowercase=False, )},
                          tokenizer=tokenizer, limit=100)
    train_test = reader.read(train_test_path)
    dev_dataset = reader.read(dev_path)
    train_dataset = []
    test_dataset = []
    for instance in train_test:
        if instance["metadata"].metadata["topic"] == TOPIC:
            test_dataset.append(instance)
        else:
            train_dataset.append(instance)
    print(len(train_dataset), len(dev_dataset), len(test_dataset))
    vocab = Vocabulary.from_instances(train_dataset + test_dataset + dev_dataset)
    embedder = BasicTextFieldEmbedder({"tokens": PretrainedTransformerEmbedder(modelname)})
    encoder = BertCLSPooler(embedder.get_output_dim())
    model = SimpleClassifier(vocab, embedder, encoder)
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    devicenum = 0
    iterator = BucketIterator([("tokens", "num_tokens")], biggest_batch_first=True)
    iterator.index_with(vocab)
    scheduler = NoamLR(optimizer, 768, 400, factor=0.015)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=3,
                      num_epochs=20,
                      cuda_device=devicenum,
                      validation_metric="+f1",
                      learning_rate_scheduler=scheduler,
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
        if i >= 12:
            topic = topic.strip()
            train_bert(train_test, dev, topic, f"/hdd/model_bert_{i}")


if __name__ == "__main__":
    main()