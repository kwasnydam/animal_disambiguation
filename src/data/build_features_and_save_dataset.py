"""Based on processed data in READ_DIRECTORY, build the features."""

import click
import os

from src.data import dataset


@click.command()
@click.argument('read_directory', type=click.Path(dir_okay=True),
                default=dataset.DEFAULT_PROCESSED_TEXT_DATA_DIRECTORY)
@click.argument('data_save_directory', type=click.Path(writable=True, dir_okay=True),
                default=dataset.DEFAULT_FEATURES_DIRECTORY)
@click.argument('data_model_save_directory', type=click.Path(writable=True, dir_okay=True),
                default=dataset.DEFAULT_DATA_MODEL_DIRECTORY)
def build_features_and_save_dataset(read_directory, data_save_directory, data_model_save_directory):
    """Build text dataset from valid sentences in READ_DIRECTORY

    Creates files 'train.npy', 'validation.npy' in DATA_SAVE_DIRECTORY and 'data_model.pickle' in
    DATA_MODEL_SAVE_DIRECTORY

    READ_DIRECTORY is directory to read processed senteces from.
        Default: <project_root>/data/processed/text
    DATA_SAVE_DIRECTORY is directory to store ready features npy file.
        Default: <project_root>/data/processed/features
    DATA_MODEL_SAVE_DIRECTORY is directory to store parameters of fitted data transformation.
        Default: <project_root>/data/processed/features
    """
    dataset.build_dataset_and_datamodel(
        read_directory=read_directory,
        data_save_directory=data_save_directory,
        data_model_save_directory=data_model_save_directory,
        vectorizer_params=dataset.DEFAULT_VECTORIZER_SETTINGS
        )
    print('Successfully built and saved dataset and data model.')

if __name__ == '__main__':
    build_features_and_save_dataset()