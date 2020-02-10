import re
import os
from typing import Set

import pandas as pd
from tqdm import tqdm


def add_articleid(claim_df: pd.DataFrame, article_df: pd.DataFrame, articles_dir_path: str) -> None:
    """
    Adds article Id columns to claim_df inplace.
    """
    ids = []
    current_topic = None
    topic_df = None
    for i, row in tqdm(claim_df.iterrows()):
        found = False
        topic = row["Topic"]
        claim_text = row["Claim corrected version"].lower()
        claim_text = re.sub("\s+", " ", claim_text)
        if topic != current_topic:
            current_topic = topic
            topic_df = article_df[article_df["Topic"] == topic]
        for j, articlerow in topic_df.iterrows():
            article_id = articlerow["article Id"]
            article_path = os.path.join(articles_dir_path, f"clean_{article_id}.txt")
            article_text = open(article_path).read().lower()
            article_text = re.sub("\s+", " ", article_text)
            if claim_text in article_text:
                found = True
                ids.append(article_id)
                break
        if not found:
            print(i)
            print(f"Nothing found for \n{topic}\n{claim_text}")
            return
    claim_df["article Id"] = ids


def get_validation_topics(motions_df: pd.DataFrame) -> Set[str]:
    validation_topics = []
    for i, row in motions_df.iterrows():
        topic = row["Topic"]
        dataset = row["Data-set"]
        if dataset != "train and test":
            validation_topics.append(topic)
    return set(validation_topics)
