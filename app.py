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
        # Extract form data safely
        form_data = request.form
        required_fields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                           "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

        # Check if all required fields are present
        if not all(field in form_data for field in required_fields):
            return jsonify({"error": "Missing input fields"}), 400

        # Convert form data to float/int and reshape for prediction
        data = [
            float(form_data["age"]),
            int(form_data["sex"]),
            int(form_data["cp"]),
            float(form_data["trestbps"]),
            float(form_data["chol"]),
            int(form_data["fbs"]),
            int(form_data["restecg"]),
            float(form_data["thalach"]),
            int(form_data["exang"]),
            float(form_data["oldpeak"]),
            int(form_data["slope"]),
            int(form_data["ca"]),
            int(form_data["thal"])
        ]
        
        # Convert to numpy array and reshape
        features = np.array(data).reshape(1, -1)
        
        # Get prediction (0 = No Disease, 1 = Disease)
        prediction = model.predict(features)[0]
        result = "Yes" if prediction == 1 else "No"

        return render_template("index.html", prediction_text=f"Heart Disease Prediction: {result}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Cloud deployment ready
