# Methods for automatic prediction of argumentation structure and stance 

## Data
The data can be acquired on the official [IBM Debater datasets page](https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml)

Data processing is described in 
**argmining/data_processing/process_ibm_multilingual_am.py**
For uniformity with the SNLI dataset the PRO/CON classes are transformed into entailment and contradition.

## Getting trained models
The trained models are located in the **paper_models** folder and get be downloaded via the `dvc pull` comand.

##  Traning and Inference
The training configs may be accesed as artifacts from the [wandb page](https://wandb.ai/hawkeoni/claim_stance?workspace=user-hawkeoni) or in the **configs** folder.

To train a model with a config use command:
```
allennlp train -s <serialization_dir> <config>
```

To get results from the model in a jsonl format use the following example:
```
allennlp predict se/model.tar.gz datasets/EviEN_relevance_test.jsonl --batch-size 8 --output-file evien_rel_zs_se.jsonl --silent --cuda-device 0
```

Metric calculation is located in **argmining/utils/evaluate.py**.