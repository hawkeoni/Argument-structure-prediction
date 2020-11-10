import json
from pathlib import Path
from collections import defaultdict

import typer
import pandas as pd


def create_df_from_jsonl(file: Path):
    columns = ["claim", "evidence", "label"]
    label_mapping = {"entailment": "Evidence", "not_entailment": "Neutral"}
    df_dict = defaultdict(list)
    for line in file.open():
        example = json.loads(line)
        df_dict["claim"].append(example["hypothesis"])
        df_dict["evidence"].append(example["premise"])
        df_dict["label"].append(label_mapping.get(example.get("label")))
    return pd.DataFrame(df_dict, columns=columns)


def main(terra_folder: Path):
    files = terra_folder.glob("*jsonl")
    for file in files:
        dump_path = Path.cwd() / "datasets" / ("terra_" + file.name.replace("jsonl", "csv"))
        create_df_from_jsonl(file).to_csv(dump_path, index=False)


if __name__ == "__main__":
    typer.run(main)
