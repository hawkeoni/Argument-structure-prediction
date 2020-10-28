from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor


@Predictor.register("topic_sentence_predictor")
class TopicSentencePredictor(Predictor):

    def _text_to_instance(self, sentence: str, topic: str = None) -> Instance:
        return self._dataset_reader.text_to_instance(sentence, topic)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict["sentence"], json_dict.get("topic"))

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def predict_texts(self, sentence: str, topic: str = None) -> JsonDict:
        instance = self._text_to_instance(sentence, topic)
        return self.predict_instance(instance)
