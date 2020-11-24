from random import choices
from typing import Tuple, List
from pathlib import Path

import pandas as pd

import typer
from tqdm import tqdm


def get_topic_text_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df[["topicText", "claims.stance", "split", "claims.claimOriginalText"]]
    df = df.rename(columns={
        "topicText": "target",
        "claims.claimOriginalText": "claim",
        "claims.stance": "stance"
        })
    train_mask = df["split"] == "train"
    df = df[["target", "claim", "stance"]]
    return df[train_mask], df[~train_mask]


def get_target_text_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df[["split", "topicTarget", "topicSentiment", "claims.claimOriginalText", "claims.stance"]]
    df["claims.stance"] = df["claims.stance"].apply(lambda x: -1 if x == "CON" else 1)
    df["stance"] = df["claims.stance"] * df["topicSentiment"]
    train_mask = df["split"] == "train"
    df = df.rename(columns={"topicTarget": "target", "claims.claimOriginalText": "claim"})
    train_df = df[train_mask][["target", "claim", "stance"]]
    test_df = df[~train_mask][["target", "claim", "stance"]]
    return train_df, test_df


def cut_string(text: str, length: int = 200) -> List[str]:
    text = text.replace("[REF]", "")
    if len(text) <= length:
        return [text]
    start = 0
    res = []
    while len(text) - start > 0:
        res.append(text[start: start + length])
        start = start + length
    if not res[-1]:
        res.pop()
    return res


def get_random_cutouts(text: str, spans: List[Tuple[int, int]]) -> List[str]:
    start = 0
    neutral = []
    for span in spans:
        ntext = text[start: span[0]]
        if ntext:
            neutral.append(ntext)
        start = span[1]
    ntext = text[start:]
    if ntext:
        neutral.append(ntext)
    return sum(map(cut_string, neutral), [])


def get_neutral_examples(stance_folder: Path, df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    gb = df.groupby("claims.article.cleanFile")
    df_dict = {"split": [], "stance": [], "target": [], "claim": []}
    for article_name, sub_df in tqdm(gb, desc="Neutral cutouts"):
        article_text = (stance_folder / article_name).read_text()
        spans = sub_df[["claims.article.cleanSpan.start", "claims.article.cleanSpan.end"]].values.tolist()
        neutral_cutouts = get_random_cutouts(article_text, spans)
        targets = sub_df.iloc[0][["topicText", "topicTarget"]].values.tolist()
        split = sub_df.iloc[0].split
        # we have neutral_cutouts and targets
        df_dict["claim"] += neutral_cutouts
        df_dict["split"] += [split] * len(neutral_cutouts)
        df_dict["stance"] += [0] * len(neutral_cutouts)
        df_dict["target"] += choices(targets, k=len(neutral_cutouts))
    neutral_df = pd.DataFrame(data=df_dict)
    train_mask = neutral_df.split == "train"
    neutral_df = neutral_df[["target", "claim", "stance"]]
    return neutral_df[train_mask], neutral_df[~train_mask]


def main(stance_folder: Path, target_claim: bool = False, text_claim: bool = False, neutral: bool = False):
    df = pd.read_csv(stance_folder / "claim_stance_dataset_v1.csv")
    df["claims.claimOriginalText"] = df["claims.claimOriginalText"].apply(lambda x: x.replace("[REF]", ""))
    train_dfs = []
    test_dfs = []
    if target_claim:
        train_df, test_df = get_target_text_dataset(df)
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    if text_claim:
        train_df, test_df = get_topic_text_dataset(df)
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    if neutral:
        train_df, test_df = get_neutral_examples(stance_folder, df)
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    train_df.to_csv(Path.cwd() / "datasets" / "ibm_cs_train_aug.csv", index=False)
    test_df.to_csv(Path.cwd() / "datasets" / "ibm_cs_test_aug.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
