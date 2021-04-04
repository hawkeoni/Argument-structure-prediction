local model_name = "bert-base-cased";
{
    "dataset_reader" : {
        "type": "NLIReaderVectorized",
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
    "train_data_path": "datasets/ArgsEN_train.jsonl",
    "validation_data_path": "datasets/ArgsEN_dev.jsonl",
    "test_data_path": "datasets/ArgsEN_test.jsonl",
    "model": {
        "type": "NLIModelVectorized",
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
        "batch_size": 32
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
                "run_name": "args_en_nli",
            }
        ],
        "num_epochs": 10,
        "validation_metric": "+accuracy"
    }
}
