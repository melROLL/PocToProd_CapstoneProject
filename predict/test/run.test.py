import unittest
from unittest.mock import MagicMock
import tempfile
import pandas as pd
import sys
import os

# Add the parent directory to the sys path to import modules from the project
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Import modules from the project
# from train.train import run
from train import run
from preprocessing.preprocessing import utils

# Mock function to load dataset for testing purposes
def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })

class TestPredict(unittest.TestCase):

    # TODO: CODE HERE
    # Mock the load_dataset function in utils module
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):
        # TODO: CODE HERE
        # Create a dictionary params for training configuration
        params = {
            "batch_size": 1,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 4,
            "verbose": 1
        }

        # TODO: CODE HERE
        # Create a temporary directory to store artifacts
        with tempfile.TemporaryDirectory() as model_dir:
            # TODO: CODE HERE
            # Run the training and get accuracy
            accuracy, _ = run.train("fake_path", params, model_dir, False)

            # TODO: CODE HERE
            # Instantiate a TextPredictionModel class from trained artifacts
            textpredictmodel = run.TextPredictionModel.from_artefacts(model_dir)

            # TODO: CODE HERE
            # Run a prediction
            predictions_obtained = textpredictmodel.predict(['php'], 2)
            print(predictions_obtained)

        # TODO: CODE HERE
        # Assert that predictions obtained have the expected shape
        self.assertGreaterEqual(predictions_obtained.shape, (1, 2))
