from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = "C:\\Users\\dhari\\Documents\\heart_disease_project\\heart_disease_model.pkl"
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
        # Get data from form
        data = [
            float(request.form["age"]),
            int(request.form["sex"]),
            int(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            int(request.form["fbs"]),
            int(request.form["restecg"]),
            float(request.form["thalach"]),
            int(request.form["exang"]),
            float(request.form["oldpeak"]),
            int(request.form["slope"]),
            int(request.form["ca"]),
            int(request.form["thal"])
        ]
        
        # Convert to numpy array and reshape
        features = np.array(data).reshape(1, -1)
        
        # Get prediction (0 = No Disease, 1 = Disease)
        prediction = model.predict(features)[0]
        result = "Yes" if prediction == 1 else "No"

        return render_template("index.html", prediction_text=f"Heart Disease Prediction: {result}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)  # Runs on http://127.0.0.1:5000/
