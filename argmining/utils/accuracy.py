import json
from argparse import ArgumentParser
from pathlib import Path

from argmining.utils.evaluate import linecount, get_predicted_class


def main(dataset_file: Path, predictions_file: Path):
    assert linecount(dataset_file) == linecount(predictions_file)
    total = 0
    correct = 0
    for d_line, p_line in zip(open(dataset_file), open(predictions_file)):
        total += 1
        d_dict = json.loads(d_line)
        p_dict = json.loads(p_line)
        true_label = d_dict["gold_label"]
        pred_label = get_predicted_class(p_dict)
        if true_label == pred_label:
            correct += 1
    if total == 0:
        print("No lines read.")
    else:
        print(f"Accuracy {correct / pred_label}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.prediction)
