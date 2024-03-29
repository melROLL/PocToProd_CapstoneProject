import unittest
from unittest.mock import MagicMock
import tempfile
import pandas as pd

# from train.train import run
from train import run
from preprocessing.preprocessing import utils

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

#
# class TestTrain_notuse(unittest.TestCase):
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    #     utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    #     def test_train_notuse(self):
    # create a dictionary params for train conf
    #     params = {
    #             "batch_size": 2,
    #             "epochs": 1,
    #             "dense_dim": 64,
    #             "min_samples_per_label": 10,
    #             "verbose": 1
    #     }

    # we create a temporary file to store artefacts
    #     with tempfile.TemporaryDirectory() as model_dir:
    # run a training
    #         accuracy, _ = run.train("fake",params,model_path="test",add_timestamp=True)

    # assert that accuracy is equal to 1.0
#     self.assertEqual(accuracy,1.0)


class TestTrain(unittest.TestCase):
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_train(self):
        # create a dictionary params for train conf
        params = {
            "batch_size": 2,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 10,
            "verbose": 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run.train("fake", params, model_path="test", add_timestamp=True)

        # Add debug prints
        print(f"Actual Accuracy: {accuracy}")

        # assert that accuracy is equal to 1.0
        self.assertEqual(accuracy, 1.0)
