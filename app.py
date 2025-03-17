from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
from src_code.baseline_model import generate_response  # Import your chatbot model

app = Flask(__name__, static_folder="", template_folder="")
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")

    response = generate_response(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

