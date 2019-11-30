from unittest import TestCase

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

from src.model import model
from src.data import dataset


class TestMMDisambiguatorBase(TestCase):
    def setUp(self):
        classifier_settings = {
            'solver': 'liblinear',
            'class_weight': 'balanced',
            'C': 1.
        }

        data_model = dataset.TextLabelsVectorizer(dataset.DEFAULT_VECTORIZER_SETTINGS)
        classifier = LogisticRegression(**classifier_settings)

        training_text_data_list = [
            'first training sentence containing word mouse on a desert',
            'second training sentence containing word mouse on a table',
            'third training sentence containing word mouse on a desk',
            'fourth training sentence containing word mouse in the mountains',
        ]
        training_text_labels_list = [
            'animal',
            'device',
            'device',
            'animal'
        ]
        training_text_data_series = pd.Series(training_text_data_list)
        training_text_labels_series = pd.Series(training_text_labels_list)

        training_features_data, training_numerical_classes = \
            data_model.fit_transform(training_text_data_series, training_text_labels_series)

        self.object_under_test = model.MMDisambiguator(data_model, classifier)

        self.training_data = {
            'text': {
                'list': {
                    'text': training_text_data_list,
                    'labels': training_text_labels_list
                },
                'series': {
                    'text': training_text_data_series,
                    'labels': training_text_labels_series
                }
            },
            'features': {
                'array': {
                    'features': training_features_data,
                    'classes': training_numerical_classes
                }
            }
        }


class TestMMDisambiguatorTrain(TestMMDisambiguatorBase):

    def test_given_arrays_of_features_and_classes_should_train_classifier(self):
        test_data = self.training_data['features']['array']
        parameters = {
            'data': test_data['features'],
            'classes': test_data['classes'],
            'report': False,
            'source': 'features'
        }
        self.object_under_test.train(**parameters)
        self.assertTrue(self.object_under_test.is_trained())

    def test_given_list_of_text_and_labels_should_train_classifier(self):
        test_data = self.training_data['text']['list']
        parameters = {
            'data': test_data['text'],
            'classes': test_data['labels'],
            'report': False,
            'source': 'text'
        }
        self.assertTrue(not self.object_under_test.is_trained())

        self.object_under_test.train(**parameters)
        self.assertTrue(self.object_under_test.is_trained())

    def test_given_pandas_series_of_text_and_labels_should_train_classifier(self):
        test_data = self.training_data['text']['series']
        parameters = {
            'data': test_data['text'],
            'classes': test_data['labels'],
            'report': False,
            'source': 'text'
        }

        self.assertTrue(not self.object_under_test.is_trained())

        self.object_under_test.train(**parameters)
        self.assertTrue(self.object_under_test.is_trained())


class TestMMDisambiguatorPredict(TestMMDisambiguatorBase):
    def fit(self):
        test_data = self.training_data['features']['array']
        parameters = {
            'data': test_data['features'],
            'classes': test_data['classes'],
            'report': False,
            'source': 'features'
        }
        self.object_under_test.train(**parameters)

    def test_if_classifier_not_trained_should_raise_NotTrainedException(self):
        self.assertTrue(not self.object_under_test.is_trained())
        with self.assertRaises(model.NotTrainedException):
            self.object_under_test.predict(self.training_data['features']['array']['features'])

    def test_given_text_list_or_pandas_series_and_correct_settings_should_work(self):
        self.fit()
        for test_data in self.training_data['text'].values():
            parameters = {
                'unseen_features': test_data['text'],
                'mode': 'classification',
                'source': 'text'
            }
            prediction = self.object_under_test.predict(**parameters)
            self.assertEqual(len(prediction), len(test_data['text']))

    def test_performance_report(self):
        self.fit()
        validation_predicted_classes = [0, 0, 1, 0]
        validation_real_classes = [0, 0, 1, 1]
        accuracy = metrics.accuracy_score(validation_real_classes, validation_predicted_classes)
        recall = metrics.recall_score(validation_real_classes, validation_predicted_classes)
        precision = metrics.precision_score(validation_real_classes, validation_predicted_classes)
        f1_score = metrics.f1_score(validation_real_classes, validation_predicted_classes)

        report = self.object_under_test.performance_report(validation_predicted_classes, validation_real_classes)
        self.assertEqual(accuracy, report['accuracy'])
        self.assertEqual(recall, report['recall'])
        self.assertEqual(precision, report['precision'])
        self.assertEqual(f1_score, report['f1'])
