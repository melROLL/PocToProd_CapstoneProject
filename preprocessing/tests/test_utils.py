import unittest
import pandas as pd
from unittest.mock import MagicMock

# Importing BaseTextCategorizationDataset from the specified module
from preprocessing.preprocessing import utils
# from preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        Test case for the _get_num_train_samples method of BaseTextCategorizationDataset.
        Uses a mock to simulate behavior and verify calculations.
        """
        # Instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # Mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # Assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        Test case for the _get_num_train_batches method of BaseTextCategorizationDataset.
        Similar to the previous test case with different method names.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        """
        Test case for the _get_num_test_batches method of BaseTextCategorizationDataset.
        """
        # Instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # Mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # Assert that _get_num_test_batches will return 100 * (1 - train_ratio) = 20
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        """
        Test case for the get_index_to_label_map method of BaseTextCategorizationDataset.
        Uses a mock to simulate behavior and verify mapping correctness.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['python', 'java', 'c#', 'javascript', 'php'])
        # Assert that the mapping is as expected
        self.assertDictEqual(base.get_index_to_label_map(),
                             {0: 'python', 1: 'java', 2: 'c#', 3: 'javascript', 4: 'php'})

    def test_index_to_label_and_label_to_index_are_identity(self):
        """
        Test case for ensuring consistency between index_to_label and label_to_index mappings.
        Uses mocks to simulate behavior and verify consistency.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # Mock get_index_to_label_map to return a predefined mapping
        base.get_index_to_label_map = MagicMock(
            return_value={0: 'python', 1: 'java', 2: 'c#', 3: 'javascript', 4: 'php'})
        # Assert that the label_to_index mapping is consistent
        self.assertDictEqual(base.get_label_to_index_map(),
                             {'python': 0, 'java': 1, 'c#': 2, 'javascript': 3, 'php': 4})

    def test_to_indexes(self):
        """
        Test case for the to_indexes method of BaseTextCategorizationDataset.
        Uses mocks to simulate behavior and verify conversion correctness.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # Mock get_label_to_index_map to return a predefined mapping
        base.get_label_to_index_map = MagicMock(
            return_value={'python': 0, 'java': 1, 'c#': 2, 'javascript': 3, 'php': 4})
        labels = ['java', 'python', 'java', 'php', 'javascript', 'c#']
        # Assert that the conversion is as expected
        self.assertListEqual(base.to_indexes(labels), [1, 0, 1, 4, 3, 2])


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        """
        Test case for the load_dataset method of LocalTextCategorizationDataset.
        Uses mocks to simulate behavior and verifies that the loaded dataset matches expectations.
        """
        # Mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # Instantiate a LocalTextCategorizationDataset and load dataset using the mocked read_csv
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # Define the expected dataset after loading
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # Confirm that the loaded dataset matches expectations
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        """
        Test case for the _get_num_samples method of LocalTextCategorizationDataset.
        Uses mocks to simulate behavior and verifies that the calculated number of samples is correct.
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 4, 5, 6],
            'tag_position': [0, 0, 0, 0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        # Instantiate a LocalTextCategorizationDataset with specific parameters
        base = utils.LocalTextCategorizationDataset("fake path", 1, 0.6, min_samples_per_label=2)
        # Verify that the calculated number of samples is as expected
        self.assertEqual(base._get_num_samples(), 3)

    def test_get_train_batch_returns_expected_shape(self):
        """
        Test case for the get_train_batch method of LocalTextCategorizationDataset.
        Uses mocks to simulate behavior and verifies that the returned batch has the expected shape.
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_a', 'tag_b', 'tag_b'],
            'tag_id': [1, 2, 3, 4, 5, 6],
            'tag_position': [0, 0, 0, 0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        # Instantiate a LocalTextCategorizationDataset with specific parameters
        base = utils.LocalTextCategorizationDataset("fake path", 2, 0.6, min_samples_per_label=2)
        # Get a train batch
        x, y = base.get_train_batch()
        # Verify that the shapes of x and y are as expected
        self.assertTupleEqual(x.shape, (2,)) and self.assertTupleEqual(y.shape, (2, 2))

    def test_get_test_batch_returns_expected_shape(self):
        """
        Test case for the get_test_batch method of LocalTextCategorizationDataset.
        Uses mocks to simulate behavior and verifies that the returned batch has the expected shape.
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_a', 'tag_b', 'tag_b'],
            'tag_id': [1, 2, 3, 4, 5, 6],
            'tag_position': [0, 0, 0, 0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        # Instantiate a LocalTextCategorizationDataset with specific parameters
        base = utils.LocalTextCategorizationDataset("fake path", 2, 0.6, min_samples_per_label=2)
        # Get a test batch
        x, y = base.get_test_batch()
        # Verify that the shapes of x and y are as expected
        self.assertTupleEqual(x.shape, (2,)) and self.assertTupleEqual(y.shape, (2, 2))

    def test_get_train_batch_raises_assertion_error(self):
        """
        Test case for ensuring that get_train_batch raises AssertionError when required conditions are not met.
        Uses mocks to simulate behavior.
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 4, 5, 6],
            'tag_position': [0, 0, 0, 0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        # Ensure that an AssertionError is raised when conditions are not met
        self.assertRaises(AssertionError, utils.LocalTextCategorizationDataset, 'fake', 3, 0.8)
