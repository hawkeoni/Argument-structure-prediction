import json
from pathlib import Path
from typing import Iterable, Dict

import typer
import pandas as pd
from tqdm import tqdm


label_mapping = {-1: "contradiction", 1: "entailment", 0: "neutral"}
save_dir = Path().cwd() / "datasets"


def transform_df(df: pd.DataFrame, column_mapping: Dict[str, str], name_prefix: str) -> Iterable[str]:
    splits = df.set.unique().tolist()
    file_mapping = {split: open(save_dir / f"{name_prefix}_{split}.jsonl", "w") for split in splits}
    for rownum, row in tqdm(df.iterrows()):
        if name_prefix == "EviEN_stance" and row["evidence_label_EN"] == 0:
            continue
        d = {}
        for k, v in column_mapping.items():
            d[v] = row[k]
        d["gold_label"] = label_mapping[d["gold_label"]]
        file = file_mapping[row["set"]]
        file.write(json.dumps(d))
        file.write("\n")
    for v in file_mapping.values():
        v.close()



def main(dataset_folder: Path):
    # Argument df processing
    argument_file = dataset_folder / "Arguments_6L_MT.csv"
    argument_df = pd.read_csv(argument_file)
    transform_df(argument_df,
                 {"topic_EN": "sentence1", "argument_EN": "sentence2", "stance_label_EN": "gold_label"},
                 "ArgsEN")
    # Evidence df stance + relevance
    evidence_df = pd.read_csv(dataset_folder / "Evidence_6L_MT.csv")
    transform_df(evidence_df,
                 {"topic_EN": "sentence1", "sentence_EN": "sentence2", "stance_label_EN": "gold_label"},
                 "EviEN_combined")
    # Evidence df stance processing
    transform_df(evidence_df,
                 {"topic_EN": "sentence1", "sentence_EN": "sentence2", "stance_label_EN": "gold_label"},
                 "EviEN_stance")
    # Evidence df relevance
    evidence_df = pd.read_csv(dataset_folder / "Evidence_6L_MT.csv")
    transform_df(evidence_df,
                 {"topic_EN": "sentence1", "sentence_EN": "sentence2", "evidence_label_EN": "gold_label"},
                 "EviEN_relevance")

if __name__ == "__main__":
    typer.run(main)
