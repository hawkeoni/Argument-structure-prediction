local model_name = "bert-base-cased";
{
    "dataset_reader" : {
        "type": "NLIReader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name
            }
        }
    },
    "train_data_path": "datasets/snli/snli_1.0_train.jsonl",
    "validation_data_path": "datasets/snli/snli_1.0_dev.jsonl",
    "test_data_path": "datasets/snli/snli_1.0_test.jsonl",
    "model": {
        "type": "NLIModel",
        "embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": model_name
                }
            }
        },
    },
    "data_loader": {
        "batch_size": 8
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 2e-5,
        },
        "epoch_callbacks": [
            {
                "type": "wandb",
                "project": "claim_stance",
                "run_name": "bert_nli",
            }
        ],
        "num_epochs": 10,
        "validation_metric": "+accuracy"
    }
}
