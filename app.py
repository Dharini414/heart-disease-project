from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Load the trained model with a relative path
model_path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Route for the home page with a form
@app.route("/")
def home():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("✅ Received request at /predict")

        # Check if request is JSON
        if request.is_json:
            data = request.json["features"]
            print("📩 Received JSON Data:", data)
        else:
            return jsonify({"error": "Invalid request format"}), 400

        # Convert to numpy array & reshape for model
        features = np.array(data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Yes" if prediction == 1 else "No"

        print(f"🔮 Prediction: {result}")
        return jsonify({"prediction": result})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Cloud deployment ready
