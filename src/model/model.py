"""Represents the classification model I am going to use in my problem

As of current version, it wraps Logistic Regression and Data Model. It has the following interface:
train - used to fit the model to the training data
predict - used to perfrom prediciton on unseen data
get_params - used to return parameters of a trained classifire
set_params - used to fill the model with a parameters of a pretrained classifier

Using sklearn.metrics it provides methods for generation of the performance report (generate_report=True)
flag on evaluation.

using Logistic Regression it trains the classifier which will be then used to make probability  predictions.
If user wishes classification instead of prediciton, he needs to pass ''

With current interface, it is possible to replace LR with another classifier as long as it complies to the same
interface (fit, predict, predict_proba, get_params, set_params)
"""
import pickle
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

from src.data import dataset

DEFAULT_CLASSIFIER_SETTINGS = {
    'solver': 'liblinear',
    'class_weight': 'balanced',
    'C': 1.
}

up = os.path.dirname
DEFAULT_ROOT_DIRECTORY = up(up(up(__file__)))   # Get directory two levels above
DEFAULT_MODEL_DIRECTORY = os.path.join(DEFAULT_ROOT_DIRECTORY, 'models')


def try_opening_file_pickle(path):
    try:
        with open(path, 'rb') as f:
            file_content = pickle.load(f)
    except FileNotFoundError as e:
        print('FileNotFound exception occured when trying to open: {}. Disambiguator build failed.'.format(
            path
        ))
        raise e
    except Exception as e:
        print('Exception occured when trying to open {}: {}'.format(path, e))
        raise e

    return file_content


def build_mmdisambiguator(data_model_params, data_model_path, classificator_parameters, classificator_path=None):
    if classificator_path is None:
        data_model = dataset.TextLabelsVectorizer(data_model_params)
        data_model_saved = try_opening_file_pickle(data_model_path)
        data_model.deserialize(data_model_saved)
        classificator = LogisticRegression(**classificator_parameters)
        disambiguator = MMDisambiguator(data_model, classificator)
    else:
        disambiguator_pieces = try_opening_file_pickle(classificator_path)
        data_model = dataset.TextLabelsVectorizer(data_model_params)
        data_model.deserialize(disambiguator_pieces['data_model'])
        classificator = disambiguator_pieces['classificator']
        disambiguator = MMDisambiguator(data_model, classificator)

    return disambiguator


class MMDisambiguator:
    """The class representing the core logic of the disambiguation app.

    It uses data_model for feature and text manipulation and Logistic Regression for performing prediction

    With 'source' flag user controls if the training/prediction is preformed from precomputed umercial features
    or text. If it is done from text, the input is put through feature_extraction first.

    Methods:
        train - fit the classifier or both data model and classifier from training data
        predict - get prediction on data. the data can be ingle and multiple samples
        performance_report - generate summary of performance
        serialize - get representation for saving
        deserialize - load serialized components
    """

    def __init__(self, data_model:dataset.TextLabelsVectorizer, classificator: LogisticRegression):
        self.data_model = data_model
        self.classificator = classificator

    def train(self, data, classes, report=False, source='features'):
        """Train the model with training data DATA and training labels CLASSES

        Args:
            data - training data (text or features)
            classes- training classes (text or numerical)
            report - flag, if True generate training report
            source - 'features': numerical, train directly. 'text': train vectorizer, transfrom, then train classifier
        """
        if source == 'text':
            features, classes = self.data_model.fit_transform(data, classes)
        else:
            features = data

        self.classificator.fit(features, classes)

        if report:
            return self.performance_report(self._classify(self.classificator.predict_proba(features)), classes)
        else:
            return None

    def transform_labels(self, labels):
        return self.data_model.transform_labels(labels)

    def predict(self, unseen_features, mode='classification', threshold=0.5, format='text', source='features'):
        """Predict classes on unseen data.

        Args:
            unseen_features - 'string' or list of 'string' if source = 'text'. feature array if source = 'features'
            mode -
                'classification' - predict probabilities and then make classifcation decision based on 'threshold
                'predicition' - return predicted probabilities
            threshold - if mode = 'classification', threshold for the decision
            source - 'text' if sentences, 'features' if input already transformed
        """
        if source == 'text':
            # print('text before transofrmation: {}'.format(len(unseen_features)))
            unseen_features = self.data_model.transform(unseen_features)
            # print('text after transofrmation: {}'.format(len(unseen_features)))

        predicted_probability = self.classificator.predict_proba(unseen_features)

        if mode == 'classification':
            classification_binary = self._classify(predicted_probability, threshold).astype(np.int)
            classification = classification_binary
            if format == 'text':
                classification = self.data_model.get_classes_name(classification_binary)

            result = []
            for idx in range(classification.shape[0]):
                result.append([classification[idx], predicted_probability[idx,classification_binary[idx]]])
            result = np.asarray(result)

        elif mode == 'prediction':
            result = predicted_probability

        return result

    def _classify(self, predicted_probabilities, threshold=0.5):
        """Decision: class based on predicted probability and threshold"""
        classes = predicted_probabilities.copy()[:,1]
        classes[classes >= threshold] = 1
        classes[classes < threshold] = 0
        return classes

    def performance_report(self, predicted_classes, real_classes):
        """Generates performance of the given classifier given predicted and real classes

        Args:
            predicted_classes: - iterable containing the prediciton results, len(num_of_samples)
            real_classes - iterable containing ground truth classes, len(num_of_samples)

        Output:
            report - dictionary containing the following fields:
                'accuracy',
                'precision',
                'recall',
                'f1_score',
                'confussion_matrix'
        """
        report = {
            'accuracy': metrics.accuracy_score(real_classes, predicted_classes),
            'precision': metrics.precision_score(real_classes, predicted_classes),
            'recall': metrics.recall_score(real_classes, predicted_classes),
            'f1': metrics.f1_score(real_classes, predicted_classes),
            'confussion_matrix': metrics.confusion_matrix(real_classes, predicted_classes, labels = [1, 0]).tolist()
        }
        return report

    def serialize(self):
        """Returns objects and parameters necessary to perform prediciton"""
        to_serialize = {
            'data_model': self.data_model.serialize(),
            'classificator': self.classificator
        }
        return to_serialize
