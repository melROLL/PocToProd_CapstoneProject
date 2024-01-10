import os
import json
import logging
import pandas as pd
import sys

# Add the parent directory to the sys path to import modules from the project
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from flask import Flask, request

# Import the TextPredictionModel from the run module
import run

# Create a Flask application
app = Flask(__name__)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Create a route to predict
@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)

    # Create a dataframe from the data
    data_df = pd.DataFrame(data)

    # Get the model path from the data
    model_path = data_df["model_path"][0]

    # Load the model
    model = run.TextPredictionModel.from_artefacts(model_path)

    # Predict the data
    predictions = model.predict(data_df["text"])

    # Create a response
    response = {"predictions": predictions}

    # Return the response as JSON
    return json.dumps(response)


# Create a route for health check
@app.route("/health", methods=["GET"])
def health():
    return "ok"


# Create a hello world route
@app.route("/", methods=["GET"])
def hello():
    return "Hello World!"


# Create a page to test the model with any text
@app.route("/test", methods=["GET"])
def test():
    return """
    <html>
        <body>
            <form action="/predict" method="post">
                <label for="text">Text:</label><br>
                <input type="text" id="text" name="text" value="How to create a new column in pandas dataframe?"><br>
                <input type="hidden" id="model_path" name="model_path" value="artefacts">
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """
