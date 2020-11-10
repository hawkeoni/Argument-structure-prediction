from pathlib import Path
from typing import Dict

from allennlp.predictors import Predictor
from flask import Flask, render_template, request, jsonify, g

from argmining.core import TopicSentencePredictor, ClaimsReader

app = Flask(__name__, template_folder="templates")
model_files = list(Path("models").glob("*tar.gz"))
model_choices = [model_file.name[:-7] for model_file in model_files]
model_predictors: Dict[str, Predictor] = \
    {
        model_name: Predictor.from_path(str(path), cuda_device=-1)
        for model_name, path in zip(model_choices, model_files)
    }


@app.route("/", methods=["get"])
def index():
    with app.app_context():
        return render_template("index.html", models=model_choices)


@app.route("/", methods=["post"])
def predict():
    model_class = request.json["model"]
    predictor = model_predictors[model_class]
    del request.json["model"]
    return jsonify(predictor.predict_json(request.json))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
