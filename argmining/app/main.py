import os
from pathlib import Path
from typing import Dict, Any

from allennlp.predictors import Predictor
from flask import Flask, render_template, request, jsonify, g

from argmining.core import TopicSentencePredictor, ClaimsReader

app = Flask(__name__, template_folder="templates")
app_dict = {}


def load_models_from_directory(directory: Path) -> Dict[str, Predictor]:
    model_files = list(Path(directory).glob("*tar.gz"))
    print(directory, model_files)
    model_choices = [model_file.name[:-7] for model_file in model_files]
    model_predictors: Dict[str, Predictor] = \
    {
        model_name: Predictor.from_path(str(path), cuda_device=-1)
        for model_name, path in zip(model_choices, model_files)
    }
    return model_predictors 


def init_app() -> Dict[str, Any]:
    dict_update = {}
    model_path = os.environ.get("MODEL_PATH")
    if model_path is None:
        raise ValueError("MODEL_PATH env variable not specified.")
    model_path = Path(model_path)
    # claim search models
    dict_update["es_models"] = load_models_from_directory(model_path / "es_models")
    dict_update["cs_models"] = load_models_from_directory(model_path / "cs_models")
    return dict_update

@app.route("/", methods=["get"])
def index():
    return render_template("index.html")
    
@app.route("/es.html", methods=["get"])
def display_es():
    model_choices = list(app_dict["es_models"].keys())
    return render_template("es.html", models=model_choices)


@app.route("/es.html", methods=["post"])
def predict_es():
    model_class = request.json["model"]
    predictor = app_dict["es_models"][model_class]
    del request.json["model"]
    return jsonify(predictor.predict_json(request.json))

@app.route("/cs.html", methods=["get"])
def display_cs():
    model_choices = list(app_dict["cs_models"].keys())
    return render_template("cs.html", models=model_choices)

@app.route("/cs.html", methods=["post"])
def predict_cs():
    model_class = request.json["model"]
    predictor = app_dict["cs_models"][model_class]
    del request.json["model"]
    return jsonify(predictor.predict_json(request.json))


if __name__ == "__main__":
    app_dict.update(init_app())
    app.run(host="0.0.0.0", port=8888)
