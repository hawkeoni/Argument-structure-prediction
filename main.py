import pandas as pd

from src import add_articleid, get_validation_topics, train_and_evaluate


def main():
    dataset_path = "datasets/IBM/CE_2015/"
    articles_path = dataset_path + "articles/"
    articles = pd.read_csv(dataset_path + "articles.txt", sep='\t')
    claims = pd.read_csv(dataset_path + "claims.txt", sep='\t')
    evidence = pd.read_csv(dataset_path + "evidence.txt", sep='\t', header=None)
    motions = pd.read_csv(dataset_path + "motions.txt", sep='\t')

    validation_topics = get_validation_topics(motions)
    validation_claims_idx = claims["Topic"].isin(get_validation_topics(motions))

    train_test_claims = claims[~validation_claims_idx]
    validation_claims = claims[validation_claims_idx]

    add_articleid(train_test_claims, articles, articles_path)
    add_articleid(validation_claims, articles, articles_path)

    train_test_topics = claims.Topic.unique()
    results = []

    for i, test_topic in enumerate(train_test_topics):
        res = train_and_evaluate(train_test_claims, validation_claims, articles_path, test_topic, f"model_{i}")
        results.append(res)
    val = 0
    for key in res.keys():
        for result in results:
            val += result[key]
        print(f"{key} = {val / (i + 1)}")
        val = 0

if __name__ == "__main__":
    main()