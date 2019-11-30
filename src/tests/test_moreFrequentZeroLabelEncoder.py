from unittest import TestCase

import pandas as pd

from src.data.dataset import MoreFrequentZeroLabelEncoder


class TestMoreFrequentZeroLabelEncoder(TestCase):
    def fit(self):
        self.object_under_test.fit(self.test_labels)

    def setUp(self):
        self.object_under_test = MoreFrequentZeroLabelEncoder()
        self.test_labels = [
            'l1',
            'l2',
            'l1'
        ]
        self.numerical_classes = [0, 1, 0]

    def test_fit(self):
        self.object_under_test.fit(self.test_labels)

        test_labels_series = pd.Series(self.test_labels)
        self.object_under_test.fit(test_labels_series)

    def test_transform(self):
        self.fit()
        self.assertTrue(all(self.object_under_test.transform(self.test_labels) == self.numerical_classes))

    def test_inverse_transform(self):
        self.fit()
        self.assertTrue(all(self.object_under_test.inverse_transform(self.numerical_classes) == self.test_labels))

    def test_get_params(self):
        self.fit()
        self.assertTrue('labels' in self.object_under_test.get_params())
        self.assertTrue('inverse_mapping' in self.object_under_test.get_params())

    def test_set_params(self):
        params = {
            'labels': {
                'l1': 0,
                'l2': 1
            },
            'inverse_mapping': {
                '0': 'l1',
                '1': 'l2'
            }
        }
        self.object_under_test.set_params(params)
        self.assertTrue(all(self.object_under_test.inverse_transform(self.numerical_classes) == self.test_labels))
        self.assertTrue(all(self.object_under_test.transform(self.test_labels) == self.numerical_classes))
