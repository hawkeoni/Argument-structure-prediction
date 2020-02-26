import re

import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from src import add_articleid, get_validation_topics, train_and_evaluate


def dump_to_file(claims_df: pd.DataFrame, articles_path: str, output_filename: str):
    f = open(output_filename, "w", encoding="utf8")
    positive = 0
    already_found = 0
    article_ids = claims_df["article Id"].unique()
    for article_id in tqdm(article_ids):
        # claims subdf - subdf with claims from certain article
        claims_subdf = claims_df[claims_df["article Id"] == article_id]
        article_text = re.sub(r"\s+", " ", open(articles_path + f"clean_{article_id}.txt").read())
        # TODO: sent tokenize to stanford parser?
        article_sentences = sent_tokenize(article_text)
        added_sentences = set()
        for i, claim_row in claims_subdf.iterrows():
            claim_text = re.sub(r"\s+", " ", claim_row["Claim corrected version"].lower())
            found = False
            for j, sentence in enumerate(article_sentences):
                if claim_text in sentence.lower():
                    if j not in added_sentences:
                        positive += 1
                        added_sentences.add(j)
                        # positive sample
                        f.write(f"{claim_row['Topic']}\t{sentence}\t1\n")
                        found = True
                    else:
                        found = True
                        already_found += 1
            if not found:
                print("Not found:\n", claim_text, "\n")
        for j, sentence in enumerate(article_sentences):
            if j not in added_sentences:
                # negative sample
                f.write(f"{claim_row['Topic']}\t{sentence}\t0\n")
    # assert positive + already_found == len(claims_df), f"{positive} {already_found} {len(claims_df)}"
    print(f"Found {positive} positive sentences of {len(claims_df)}, with {already_found} in duplicate sentences.")

dataset_path = "datasets/IBM/CE_2015/"
articles_path = dataset_path + "articles/"
articles = pd.read_csv(dataset_path + "articles.txt", sep='\t')
claims = pd.read_csv(dataset_path + "claims.txt", sep='\t')
motions = pd.read_csv(dataset_path + "motions.txt", sep='\t')
train_test_topics = motions[motions["Data-set"] == "train and test"]["Topic"]
assert len(train_test_topics) == 39
f = open("train_test_topics.txt", "w")
for topic in train_test_topics:
    f.write(topic)
    f.write("\n")
f.close()
validation_topics = get_validation_topics(motions)
validation_claims_idx = claims["Topic"].isin(validation_topics)

train_test_claims = claims[~validation_claims_idx]
validation_claims = claims[validation_claims_idx]

add_articleid(train_test_claims, articles, articles_path)
add_articleid(validation_claims, articles, articles_path)
dump_to_file(train_test_claims, articles_path, "train_test.txt")
dump_to_file(validation_claims, articles_path, "val.txt")
