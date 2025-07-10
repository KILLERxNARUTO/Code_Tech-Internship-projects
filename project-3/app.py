# Import necessary libraries
from flask import Flask, request, render_template
import pickle  # For loading the saved machine learning model
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained ML model from disk using pickle
model = pickle.load(open("model.pkl", "rb"))

# Route for the home page
@app.route('/')
def home():
    # Renders the main page with the input form
    return render_template('index.html')

# Route to handle prediction logic after form submission
@app.route('/predict', methods=["POST"])
def predict():
    # Extract input values from the form and convert to float
    features = [float(x) for x in request.form.values()]

    # Make a prediction using the loaded model
    prediction = model.predict([features])[0]

    # Render the same page with the prediction result
    return render_template("index.html", prediction_text=f"Predicted Iris Class: {prediction}")

# Run the app (in debug mode for development)
if __name__ == "__main__":
    app.run(debug=True)
