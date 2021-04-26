import json
from pathlib import Path
from typing import Union

import typer
import pandas as pd
from tqdm import tqdm


def remove_ref(s: str) -> str:
    return s.replace("[REF]", "")

def get_label(label: Union[int, str]):
    if isinstance(label, str):
        return label
    return {0: "Neutral", 1: "Evidence"}[label]

def process_ibm_evidencesearch_df(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Remove refs, improve label.
    """
    f = open(filename, "w")
    for rownum, row in tqdm(df.iterrows()):
        d = {}
        d["sentence1"] = remove_ref(row["topic"])
        d["sentence2"] = remove_ref(row["candidate"])
        d["gold_label"] = get_label(row["label"])
        f.write(json.dumps(d))
        f.write("\n")
    f.close()



def main(evidence_search_folder: Path):
    save_path = Path.cwd() / "datasets"
    # English
    print("Eng train")
    train_df = pd.read_csv(evidence_search_folder / "train.csv")
    print("Eng test")
    test_df = pd.read_csv(evidence_search_folder / "test.csv")
    process_ibm_evidencesearch_df(train_df, save_path / "ibm_es_train.jsonl")
    process_ibm_evidencesearch_df(test_df, save_path / "ibm_es_test.jsonl")
    # Russian
    print("Ru train")
    train_df = pd.read_csv(evidence_search_folder / "ibm_es_rutrain.csv")
    print("Ru test")
    test_df = pd.read_csv(evidence_search_folder / "test.csv")
    test_df = pd.read_csv(evidence_search_folder / "ibm_es_rutest.csv")
    process_ibm_evidencesearch_df(train_df, save_path / "ibm_es_rutrain.jsonl")
    process_ibm_evidencesearch_df(test_df, save_path / "ibm_es_rutest.jsonl")

if __name__ == "__main__":
    typer.run(main)
