from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the trained MLP model
model = tf.keras.models.load_model("animal_recommender_model.h5")

# List of animals
animals = ['Dog', 'Cat', 'Dinosaur', 'Horse', 'Rabbit', 'Turtle', 'Goat', 'Deer', 'Panda', 'Koala']

# Create Flask app
app = Flask(__name__)

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict_animal():
    data = request.json  # Get input data as JSON
    # Extract input features
    input_data = np.array([
        [
            data.get("activity_level", 0),
            data.get("space_available", 0),
            data.get("care_level", 0),
            data.get("friendliness", 0)
        ]
    ])
    # Make a prediction
    prediction = model.predict(input_data)
    predicted_animal = animals[np.argmax(prediction)]
    return jsonify({"recommended_animal": predicted_animal})  # Return result

# Home route to check API status
@app.route("/", methods=["GET"])
def home():
    return "Animal Recommendation API is running!"

# Run the app locally
if __name__ == "__main__":
    app.run()
