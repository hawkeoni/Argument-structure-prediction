from pathlib import Path

import typer
import pandas as pd


def remove_ref(s: str) -> str:
    return s.replace("[REF]", "")


def process_ibm_evidencesearch_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove refs, improve label.
    """
    df["claim"] = df["topic"].apply(remove_ref)
    df["evidence"] = df["candidate"].apply(remove_ref)
    df["label"] = df["label"].apply(lambda x: "Evidence" if x == 1 else "Neutral")
    columns_to_keep = ["claim", "evidence", "label"]
    return df[columns_to_keep]


def main(evidence_search_folder: Path):
    train_df = pd.read_csv(evidence_search_folder / "train.csv")
    test_df = pd.read_csv(evidence_search_folder / "test.csv")
    train_df = process_ibm_evidencesearch_df(train_df)
    test_df = process_ibm_evidencesearch_df(test_df)
    train_df.to_csv(Path.cwd() / "datasets/ibm_es_train.csv", index=False)
    test_df.to_csv(Path.cwd() / "datasets/ibm_es_test.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
