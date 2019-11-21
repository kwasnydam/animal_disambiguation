import click
import os
import json
import pickle

import numpy as np

from src.data import dataset
from src.model import model


@click.command()
@click.argument('train_data_read_path', type=click.Path(dir_okay=False),
                default=os.path.join(dataset.DEFAULT_FEATURES_DIRECTORY,'train.npy'))
@click.argument('data_model_read_path', type=click.Path(dir_okay=False),
                default=os.path.join(dataset.DEFAULT_DATA_MODEL_DIRECTORY, 'data_model.pickle'))
@click.argument('data_model_params',
                default=dataset.DEFAULT_VECTORIZER_SETTINGS)
@click.argument('model_save_directory', type=click.Path(writable=True, dir_okay=True),
                default=model.DEFAULT_MODEL_DIRECTORY)
@click.argument('model_params',
                default=model.DEFAULT_CLASSIFIER_SETTINGS)
@click.argument('save_training_report', type=bool,
                default=True)
@click.argument('training_report_save_path', type=click.Path(writable=True, dir_okay=False),
                default=os.path.join(model.DEFAULT_MODEL_DIRECTORY, 'training_report.json'))
def train_and_save_model(
        train_data_read_path,
        data_model_read_path,
        data_model_params,
        model_save_directory,
        model_params,
        save_training_report,
        training_report_save_path
):
    """Build classification model with parameters MODEL_PARAMS and trains it with data from TRAIN_DATA_READ_PATH

    Creates files 'trained_model.pickle' containing the trained classifier ready to be used on evaluation.
    If SAVE_TRAINING_REPORT, then creates training_report.csv in models directory, which contains information
    about trained classifier parameters and performance scores

    TRAIN_DATA_READ_PATH is path to read processed features from.
        Default: <project_root>/data/processed/features/train.npy
    DATA_MODEL_READ_PATH is path to read the vectorizer parameters to vectorize new texts.
        Default: <project_root>/models/data_model.pickle
    MODEL_SAVE_DIRECTORY is directory to store trained classifier and it's parameters.
        Default: <project_root>/models/
    MODEL_PARAMS is a dictionary of {parameter:value} pairs to build a classifier
        Default: check model.DEFAULT_CLASSIFIER_SETTINGS
    SAVE_TRAINING_REPORT is a flag of weather to generate and save the report from the training
        Default: True
    TRAINING_REPORT_SAVE_PATH
        Default <project_root>/models/training_report.csv
    """

    pred_model = model.build_mmdisambiguator(
        data_model_params=data_model_params,
        data_model_path=data_model_read_path,
        classificator_parameters=model_params
    )

    train_data = np.load(train_data_read_path, allow_pickle=True)
    features, classes = train_data[:, :-1], train_data[:, -1]

    report = pred_model.train(features=features, classes=classes, report=save_training_report)
    trained_model_parameters = pred_model.get_classifier_params()
    if report is not None:
        print(report)
        with open(training_report_save_path, 'w') as rf:
            json.dump(report, rf)

    model_save_path = os.path.join(model_save_directory, 'trained_model.pickle')
    with open(model_save_path, 'wb') as mf:
        pickle.dump(pred_model, mf)
    model_parms_save_path = os.path.join(model_save_directory, 'model_params.json')
    with open(model_parms_save_path, 'w') as mpf:
        json.dump(trained_model_parameters, mpf)


if __name__ == '__main__':
    train_and_save_model()