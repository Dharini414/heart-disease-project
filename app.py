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
        # Extract JSON data (since jQuery is sending JSON)
        input_data = request.get_json()

        # Debugging: Print received data
        print("📩 Received JSON Data:", input_data)

        # Ensure 'features' key exists in JSON
        if "features" not in input_data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        # Convert all values to float
        features = np.array([float(x) for x in input_data["features"]]).reshape(1, -1)

        # Debugging: Print converted features
        print("✅ Converted Features:", features)

        # Get prediction (0 = No Disease, 1 = Disease)
        prediction = model.predict(features)[0]
        result = "Yes" if prediction == 1 else "No"

        return jsonify({"prediction": result})

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Cloud deployment ready
