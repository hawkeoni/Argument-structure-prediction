from allennlp.predictors import Predictor
from flask import Flask, render_template, request

import src

app = Flask(__name__, template_folder="templates")
predictor = Predictor.from_path("model.tar.gz")

@app.route("/topic_claim", methods=["get", "post"])
def topic_claim():
    if request.method == "GET":
        return render_template("topic_claim.html")
    elif request.method == "POST":
        evidence = request.form.get('evidence')  # запрос к данным формы
        topic = request.form.get('topic')
        return str(predictor.predict_texts(sentence=evidence, topic=topic))



if __name__ == "__main__":
    app.run()
