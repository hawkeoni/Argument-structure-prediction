from allennlp.common import JsonDict
from allennlp.data import Instance
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