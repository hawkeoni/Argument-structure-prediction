import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict, Counter

import typer
from numpy import argmax
from tqdm import tqdm


def linecount(file: Path) -> int:
    i = 0
    for _ in open(file):
        i += 1
    return i


def get_predicted_class(pred: Dict, ignored_labels: List[str]) -> str:
    labels = pred["labels"]
    logits = pred["logits"]
    for ignored_label in ignored_labels:
        logits[labels.index(ignored_label)] = -float("inf")
    return labels[argmax(logits)]


def get_metric(metric: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = metric["tp"], metric["fp"], metric["fn"]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return {"f1": fscore, "precision": precision, "recall": recall}


def main(dataset: Path, predictions: Path, ignored_labels: List[str] = []):
    assert linecount(dataset) == linecount(predictions), "File line counts differ!"
    metrics = defaultdict(Counter)
    for dataset_line, pred_line in tqdm(zip(open(dataset), open(predictions))):
        data = json.loads(dataset_line)
        pred = json.loads(pred_line)
        true_label = data["gold_label"]
        topic = data["sentence1"]
        pred_label = get_predicted_class(pred, ignored_labels)
        if pred_label == "entailment" and true_label == "entailment":
            metrics[topic]["tp"] += 1
        elif pred_label == "entailment" and true_label == "contradiction":
            metrics[topic]["fp"] += 1
        elif pred_label == "contradiction" and true_label == "entailment":
            metrics[topic]["fn"] += 1
        elif pred_label == "contradiction" and true_label == "contradiction":
            metrics[topic]["tn"] += 1

    micro = defaultdict(int)
    macro = {"f1": 0, "precision": 0, "recall": 0}
    # metrics - topic -> tp, fp, fn
    # metric -> tp, fp, fn
    # micro - tp, fp, fn
    # calc_metric - prec, rec, f1
    # macro -> prec, rec, f1
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for topic, metric in metrics.items():
        for k, v in metric.items():
            micro[k] += v
        calc_metric = get_metric(metric)
        print(topic)
        print(calc_metric)
        tp += metric["tp"]
        tn += metric["tn"]
        fp += metric["fp"]
        fn += metric["fn"]
        for k, v in calc_metric.items():
            macro[k] += v

    print("micro")
    print(get_metric(micro))
    print("macro")
    for k, v in macro.items():
        macro[k] = v / len(metrics)
    print(macro)
    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))


if __name__ == "__main__":
    typer.run(main)
