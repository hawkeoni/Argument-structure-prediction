import json
from pathlib import Path
from typing import Iterable

import typer
import pandas as pd
from tqdm import tqdm


def transform_df_to_json(df: pd.DataFrame) -> Iterable[str]:
    # yield self.text_to_instance(d["sentence1"], d["sentence2"], d["gold_label"])
    # "claims.claimOriginalText"
    left_mapping = {"PRO": 1, "CON": -1}
    label_mapping = {-1: "contradiction", 1: "entailment", 0: "neutral"}
    for rownum, row in tqdm(df.iterrows()):
        d = {}
        d["sentence1"] = row["topicText"]
        d["sentence2"] = row["claims.claimOriginalText"]
        table_stance = left_mapping[row["claims.stance"]]
        calc_stance = row["topicSentiment"] * row["claims.targetsRelation"] * row["claims.claimSentiment"]
        if table_stance == calc_stance:
            d["gold_label"] = label_mapping[table_stance]
            yield json.dumps(d), row["split"]


def main(stance_folder: Path):
    save_dir = Path().cwd() / "datasets"
    train_file = save_dir / "ibm_cs_train.jsonl"
    test_file = save_dir / "ibm_cs_test.jsonl"
    train_file = open(train_file, "w")
    test_file = open(test_file, "w")
    df = pd.read_csv(stance_folder / "claim_stance_dataset_v1.csv")
    df = df.dropna()
    file_map = {"train": train_file, "test": test_file}
    for jsonl, split in transform_df_to_json(df):
        file_map[split].write(jsonl)
        file_map[split].write("\n")
    train_file.close()
    test_file.close()


if __name__ == "__main__":
    typer.run(main)
