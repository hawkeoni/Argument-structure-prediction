from typing import List
from allennlp.models import Model
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.predictors import Predictor


@Predictor.register("topic_sentence_predictor")
class TopicSentencePredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(**json_dict)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        result = self.predict_instance(instance)
        result["positive_prob"] = result["probs"][self._model.positive_label]
        return result


@Predictor.register("claim_stance_predictor")
class ClaimStancePredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(**json_dict)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        result = self.predict_instance(instance)
        return result


@Predictor.register("NLIPredictor")
class NLIPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True, neutral: bool = True) -> None:
        super().__init__(model, dataset_reader, frozen)
        self.neutral = neutral
        self.label2idx = self._model.vocab.get_token_to_index_vocabulary("labels")
        self.idx2label = self._model.vocab.get_index_to_token_vocabulary("labels")
        self.labels = list(self.label2idx.keys())

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict["sentence1"],
                                                     json_dict["sentence2"],
                                                     json_dict.get("gold_label"))

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        result = self.predict_instance(instance)
        result["labels"] = self.labels
        result["scores"] = {k: v for k, v in zip(self.labels, result["logits"])}
        return result
 
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            output["labels"] = self.labels
            output["scores"] = {k: v for k, v in zip(self.labels, output["logits"])}
        return sanitize(outputs)