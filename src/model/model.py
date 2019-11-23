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
    data_model = try_opening_file_pickle(data_model_path)

    classificator = LogisticRegression(**classificator_parameters)
    disambiguator = MMDisambiguator(data_model, classificator)
    if classificator_path is not None:
        disambiguator = try_opening_file_pickle(classificator_path)

    return disambiguator


class MMDisambiguator:

    def __init__(self, data_model:dataset.TextLabelsVectorizer, classificator: LogisticRegression):
        self.data_model = data_model
        self.classificator = classificator

    def train(self, features, classes, report=False):
        self.classificator.fit(features, classes)
        if report:
            return self.performance_report(self._classify(self.classificator.predict_proba(features)), classes)
        else:
            return None

    def predict(self, unseen_features, mode='classification', threshold=0.5, format='text', report=False):
        predicted_probability = self.classificator.predict_proba(unseen_features)

        if mode == 'classification':
            classification_binary = self._classify(predicted_probability, threshold).astype(np.int)
            classification = classification_binary
            if format == 'text':
                classification = self.data_model.get_classes_name(classification_binary)
                # print(classification)

            result = []
            for idx in range(classification.shape[0]):
                result.append([classification[idx], predicted_probability[idx,classification_binary[idx]]])
            result = np.asarray(result)
        elif mode == 'prediction':
            result = predicted_probability
        return result

    def _classify(self, predicted_probabilities, threshold=0.5):
        classes = predicted_probabilities.copy()[:,1]
        classes[classes >= threshold] = 1
        classes[classes < threshold] = 0
        return classes

    def performance_report(self, predicted_classes, real_classes):
        report = {
            'accuracy': metrics.accuracy_score(real_classes, predicted_classes),
            'precision': metrics.precision_score(real_classes, predicted_classes),
            'recall': metrics.recall_score(real_classes, predicted_classes),
            'f1': metrics.f1_score(real_classes, predicted_classes),
            'confussion_matrics': metrics.confusion_matrix(real_classes, predicted_classes, labels = [1, 0]).tolist()
        }
        return report

    def get_classifier_params(self):
        return self.classificator.get_params()

    def set_classifier_params(self, params):
        self.classificator.set_params(**params)
