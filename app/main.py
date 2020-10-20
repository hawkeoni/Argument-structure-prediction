from flask import Flask, render_template


app = Flask(__name__, template_folder="templates")


@app.route("/topic_claim")
def topic_claim():
    return render_template("topic_claim.html")


if __name__ == "__main__":
    app.run()
