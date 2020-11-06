from allennlp.predictors import Predictor
from flask import Flask, render_template, request, jsonify

from argmining.core import TopicSentencePredictor, ClaimsReader

app = Flask(__name__, template_folder="templates")
predictor = Predictor.from_path("newtmp/model.tar.gz")


@app.route("/", methods=["get", "post"])
def topic_claim():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        print(request.form)
        pred = predictor.predict_json(request.form)
        return jsonify(pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
