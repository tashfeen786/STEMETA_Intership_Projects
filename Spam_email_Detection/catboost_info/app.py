from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)

# Load the saved CatBoost model
with open("F:\\STEMETA_Intership_Projects\\Spam_email_Detection\\catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get input values from the form and convert them to floats
            features = [
                float(request.form["feature1"]),
                float(request.form["feature2"]),
                float(request.form["feature3"]),
                # Add more form fields here if needed
            ]
            
            # Reshape the features for prediction
            prediction = model.predict([features])[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

# catboost_info/app.py
# This file contains the Flask application code for serving the CatBoost model.
