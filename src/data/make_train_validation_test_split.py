"""Based on the PROCESSED TEXT files in the READ_DIRECTORY, split into train-valid-test sets.

This script is meant for usage after the 'build_text_dataset.py'.
"""

import click
import codecs
import os

import src.data.dataset as dataset


@click.command()
@click.argument('read_path', type=click.Path(dir_okay=False),
                default=os.path.join(
                dataset.DEFAULT_PROCESSED_TEXT_DATA_DIRECTORY, 'dataset.csv'))
@click.argument('save_directory', type=click.Path(writable=True, dir_okay=True),
                default=dataset.DEFAULT_PROCESSED_TEXT_DATA_DIRECTORY)
@click.argument('encoding', default=dataset.DEFAULT_ENCODING)
def split_data(read_path, save_directory, encoding='utf-8'):
    """Split text dataset into train/validation/test from data in READ_DIRECTORY

    READ_DIRECTORY is directory to read processed text data from.
        Default: <project_root>/data/raw
    SAVE_DIRECTORY is directory to store splitted csv files.
        Default: <project_root>/data/processed
    ENCODING is the encoding used to save the dataset
        Default: 'utf-8'

    Creates files 'train.csv', 'validation.csv', 'test.csv' in SAVE_DIRECTORY
    """
    dataset.train_validation_test_split(read_path, save_directory, encoding)


if __name__ == '__main__':
    split_data()
