import json
from pathlib import Path
from typing import Iterable, Dict

import typer
import pandas as pd
from tqdm import tqdm


label_mapping = {0: "Not Relevant", 1: "Relevant"}
# label_mapping = {0: "neutral", 1: "entailment"}
save_dir = Path().cwd() / "relevance_datasets"
save_dir.mkdir(exist_ok=True)


def transform_df(
    df: pd.DataFrame, column_mapping: Dict[str, str], name_prefix: str
) -> Iterable[str]:
    splits = df.set.unique().tolist()
    file_mapping = {
        split: open(save_dir / f"{name_prefix}_{split}.jsonl", "w") for split in splits
    }
    for rownum, row in tqdm(df.iterrows()):
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
    # Evidence df relevance
    evidence_df = pd.read_csv(dataset_folder / "Evidence_6L_MT.csv")
    transform_df(
        evidence_df,
        {
            "topic_EN": "sentence1",
            "sentence_EN": "sentence2",
            "evidence_label_EN": "gold_label",
        },
        "EviEN_relevance",
    )


if __name__ == "__main__":
    typer.run(main)
