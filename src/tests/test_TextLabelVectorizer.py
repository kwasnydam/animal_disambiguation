import unittest

from src.data.dataset import TextLabelsVectorizer
from sklearn.exceptions import NotFittedError
import numpy as np
import pandas as pd


DEFAULT_VECTORIZER_SETTINGS = {
    'ngram_range': (1, 3),
    'min_df': 2,
    'max_df': 1.0
}


class TestTextLabelVectorizerBase(unittest.TestCase):

    def setUp(self):
        self.default_params = {
            'ngram_range': (1, 3),
            'min_df': 2,
            'max_df': 1.0
        }

    def tearDown(self):
        del self


class TestTextLabelVectorizerInstantiation(TestTextLabelVectorizerBase):

    def test_instantiation_with_default_parameters_should_work(self):
        instance = TextLabelsVectorizer(self.default_params)

    def test_instantiation_with_incorrect_parameters_should_not_work(self):
        wrong_params = {
            'wrong param 1': 'wrong param 1 value'
        }
        with self.assertRaises(Exception):
            instance = TextLabelsVectorizer(wrong_params)


class TestTextLabelVectorizerFitting(TestTextLabelVectorizerBase):

    def setUp(self):
        super(TestTextLabelVectorizerFitting, self).setUp()
        self.object_under_test = TextLabelsVectorizer(self.default_params)

    def test_tokenize_given_sentence_string_returns_list_of_tokens(self):
        test_sentence = 'this is a test sentence'
        self.assertIsInstance(self.object_under_test.tokenize(test_sentence), list)

    def test_fit_given_list_of_sentences_should_fit_vectorizer(self):
        test_list_of_sentences = [
            'this is sentence 1',
            'and this is sentence 2'
        ]
        test_list_of_labels = [
            'label_1',
            'label_2'
        ]
        self.object_under_test.fit(test_list_of_sentences, test_list_of_labels)
        self.assertTrue(self.object_under_test.is_fitted())

    def test_fit_given_pandas_dataframe_of_sentences_should_fit_vectorizer(self):
        test_list_of_sentences = [
            'this is sentence 1',
            'and this is sentence 2'
        ]
        test_list_of_labels = [
            'label_1',
            'label_2'
        ]
        test_df = pd.DataFrame(list(zip(test_list_of_sentences, test_list_of_labels)))
        self.object_under_test.fit(test_df.iloc[:,0], test_df.iloc[:,1])
        self.assertTrue(self.object_under_test.is_fitted())

    def test_fit_given_non_list_should_throw_type_error(self):
        test_sentence = 'I am not a list'
        test_labels = ['label_1']
        with self.assertRaises(TypeError):
            self.object_under_test.fit(test_sentence, test_labels)

    def test_fit_given_empty_list_on_either_argument_should_throw_value_error(self):
        test_sentence = []
        test_labels = ['label_1']
        with self.assertRaises(ValueError):
            self.object_under_test.fit(test_sentence, test_labels)

        test_sentence = ['sentence 1']
        test_labels = []
        with self.assertRaises(ValueError):
            self.object_under_test.fit(test_sentence, test_labels)

    def test_fit_labels_given_list_of_labels_should_assign_0_to_more_frequent_label(self):
        test_list_of_labels = [
            'label_1',
            'label_2',
            'label_2'
        ]
        expected_output = [1, 0, 0]
        self.object_under_test.fit_labels(test_list_of_labels)
        encoded_labels = self.object_under_test.transform_labels(test_list_of_labels)
        self.assertTrue(all(encoded_labels == expected_output))

    def test_fit_labels_given_list_or_pandas_series_should_work(self):
        test_list_of_labels = [
            'label_1',
            'label_2',
            'label_2'
        ]
        test_array_of_labels = pd.Series(test_list_of_labels)
        self.object_under_test.fit_labels(test_list_of_labels)
        self.object_under_test.fit_labels(test_array_of_labels)

    def test_transform_on_not_fitted_vectorizer_throws_NotFittedError_exception(self):
        test_list_of_sentences = [
            'this is sentence 1',
            'and this is sentence 2'
        ]
        with self.assertRaises(NotFittedError):
            self.object_under_test.transform(test_list_of_sentences)


class TestTextLabelVectorizerTransforming(TestTextLabelVectorizerBase):

    def setUp(self):
        super(TestTextLabelVectorizerTransforming, self).setUp()
        self.correct_test_sentences = [
            'this is sentence 1',
            'and this is sentence 2',
            'sentence 2 again'
        ]
        self.correct_test_lables = [
            'sentence_1',
            'sentence_2',
            'sentence_2'
        ]
        self.object_under_test = TextLabelsVectorizer(self.default_params)
        self.object_under_test.fit(self.correct_test_sentences, self.correct_test_lables)

    def test_transform_given_list_of_2_sentences_returns_numpy_array_with_2_rows(self):
        sentences_to_transform = [
            'evaluation sentence 1',
            'evaluation sentence 2'
        ]
        transformed_sentences = self.object_under_test.transform(sentences_to_transform)
        self.assertIsInstance(transformed_sentences, np.ndarray)
        self.assertEqual(transformed_sentences.shape[0], len(sentences_to_transform))

    def test_transform_labels_should_assign_0_to_more_frequent_class(self):
        more_frequent = 'sentence_2'
        self.assertEqual(self.object_under_test.transform_labels(more_frequent), 0)

    def test_get_classes_name_should_return_label_from_numerical_representation(self):
        self.assertEqual(self.object_under_test.get_classes_name(0), 'sentence_2')
        self.assertEqual(self.object_under_test.get_classes_name(1), 'sentence_1')

    def test_serialize(self):
        serialized_data_model = self.object_under_test.serialize()
        self.assertTrue('label_encoder' in serialized_data_model)
        self.assertTrue('vectorizer' in serialized_data_model)

if __name__ == '__main__':
    unittest.main()
