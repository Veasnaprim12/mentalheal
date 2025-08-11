# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import os
import sys
import logging

# Configure basic logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the Flask application instance
app = Flask(__name__)

# Define the paths to the model and vectorizer files
# Using os.path.join for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer1.pkl")

# Load the machine learning model and vectorizer
# Use a try...except block to handle missing files and exit gracefully
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("XGBoost model loaded successfully from %s", MODEL_PATH)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logging.info("Vectorizer loaded successfully from %s", VECTORIZER_PATH)

except FileNotFoundError as e:
    logging.critical(f"Error: A required file was not found. Please ensure both '{MODEL_PATH}' and '{VECTORIZER_PATH}' are in the same directory as this script.")
    logging.critical("Application cannot start without the model and vectorizer. Exiting.")
    sys.exit(1)

# Map numeric prediction indices to category names
label_map = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality disorder",
    5: "Stress",
    6: "Suicidal"
}

# --- New Routes for Home Page and Chatbot Page ---
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests from the frontend, predicts the category of the
    user's message, and returns the result as JSON.
    """
    data = request.get_json()
    
    # Validate the incoming data
    if not data or "message" not in data:
        logging.warning("Received a prediction request with no 'message' field.")
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]
    logging.info("Predicting for user message: '%s'", user_message)

    # Transform the user message using the loaded vectorizer
    X = vectorizer.transform([user_message])

    # Make a prediction with the model
    prediction_idx = model.predict(X)[0]
    prediction_label = label_map.get(int(prediction_idx), "Unknown")
    
    logging.info("Prediction result: %s", prediction_label)

    # Return the prediction as a JSON response
    # The frontend expects a key named 'response'
    return jsonify({"response": "Ohh now you are " + prediction_label})

@app.route("/health", methods=["GET"])
def health_check():
    """A simple endpoint to check if the server is running."""
    logging.info("Health check endpoint accessed.")
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True)
