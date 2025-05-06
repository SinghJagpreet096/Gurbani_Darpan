from flask import Flask, request, jsonify
from flask_cors import CORS
# from model import Model
from config import Config
from chat import Model

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication
model_name = Config().model
m = Model(model_name)

@app.route("/response", methods=["POST"])
def response():
    data = request.json
    user_input = data.get("text", "")
    print("user input", user_input)
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    response_text = m.response(user_input)  # Simple response logic
    print("response generated", response_text)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)