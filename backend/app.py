from flask import Flask, request, jsonify
from flask_cors import CORS
from model import Model

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication
m = Model("llama3.2")

@app.route("/response", methods=["POST"])
def response():
    data = request.json
    user_input = data.get("text", "")
    response_text = m.generate(user_input)  # Simple response logic
    print("response generated", response_text)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
