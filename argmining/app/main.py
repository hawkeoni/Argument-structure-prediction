from allennlp.predictors import Predictor
from flask import Flask, render_template, request

from argmining.core import TopicSentencePredictor, ClaimsReader

app = Flask(__name__, template_folder="templates")
predictor = Predictor.from_path("model2/model.tar.gz")


@app.route("/topic_claim", methods=["get", "post"])
def topic_claim():
    if request.method == "GET":
        return render_template("topic_claim.html")
    elif request.method == "POST":
        evidence = request.form.get('evidence')  # запрос к данным формы
        topic = request.form.get('topic')
        query = {"sentence": evidence, "topic": topic}
        result = predictor.predict_json(query)
        evidence_prob = result["probs"][1]
        return f"Probability of evidence entailing topic: {evidence_prob}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
