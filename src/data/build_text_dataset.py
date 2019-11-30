"""Based on the FILTERED files in the READ_DIRECTORY, build the dataset.

This script is meant for usage after the 'download_raw_data'.
"""

import click
import codecs
import os

import src.data.dataset as dataset


@click.command()
@click.argument('read_directory', type=click.Path(dir_okay=True),
                default=dataset.DEFAULT_INTERIM_DATA_DIRECTORY)
@click.argument('save_directory', type=click.Path(writable=True, dir_okay=True),
                default=dataset.DEFAULT_PROCESSED_TEXT_DATA_DIRECTORY)
@click.argument('encoding', default=dataset.DEFAULT_ENCODING)
def filter_raw_data(read_directory, save_directory, encoding='utf-8'):
    """Build text dataset from valid sentences in READ_DIRECTORY

    READ_DIRECTORY is directory to read filtered/normalized data from.
        Default: <project_root>/data/raw
    SAVE_DIRECTORY is directory to store ready dataset csv file.
        Default: <project_root>/data/interim
    ENCODING is the encoding used to save the dataset
        Default: 'utf-8'

    Creates files 'dataset.csv' in SAVE_DIRECTORY
    """
    dataset.build_text_dataset(read_directory, save_directory, encoding)


if __name__ == '__main__':
    filter_raw_data()
