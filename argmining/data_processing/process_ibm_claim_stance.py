from pathlib import Path

import pandas as pd

import typer


def main(stance_folder: Path):
    df = pd.read_csv(stance_folder / "claim_stance_dataset_v1.csv")
    df = df[["split", "topicTarget", "topicSentiment", "claims.claimOriginalText", "claims.stance"]]
    df["claims.claimOriginalText"] = df["claims.claimOriginalText"].apply(lambda x: x.replace("[REF]", ""))
    df["claims.stance"] = df["claims.stance"].apply(lambda x: -1 if x == "CON" else 1)
    df["stance"] = df["claims.stance"] * df["topicSentiment"]
    train_mask = df["split"] == "train"
    df = df.rename(columns={"topicTarget": "target", "claims.claimOriginalText": "claim"})
    train_df = df[train_mask][["target", "claim", "stance"]]
    test_df = df[~train_mask][["target", "claim", "stance"]]
    train_df.to_csv(Path.cwd() / "datasets" / "ibm_cs_train.csv", index=False)
    test_df.to_csv(Path.cwd() / "datasets" / "ibm_cs_test.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
