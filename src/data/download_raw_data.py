"""Use this script to download raw data from wikipedia.

You can specify custom TITLES from wikipedia to be downloaded, check dataset.DEFAULT_TITLES for the format.
"""

import click

import src.data.dataset as dataset


@click.command()
@click.argument('titles', default=dataset.DEFAULT_TITLES)
@click.argument('save_directory', type=click.Path(writable=True, dir_okay=True),
                default=dataset.DEFAULT_RAW_DATA_DIRECTORY)
@click.argument('encoding', default=dataset.DEFAULT_ENCODING)
def download_raw_data(titles, save_directory, encoding='utf-8'):
    """Download TITLES content from wikipedia and save to SAVE_DIRECTORY with ENCODING

    TITLES is a dictionary of {context: (titles)} pairs, for example {animal: ('mouse)}, to access in wikipedia..
    Make sure not to provide ambigous terms, as it may throw an exception when attempting to download the content
        Default:
            {
                'animal': ('mouse', 'kangaroo mouse', 'hopping mouse'),
                'device': ('computer mouse', 'optical mouse')
            }
    SAVE_DIRECTORY is directory to store downloaded raw data.
        Default: <project_root>/data/raw
    ENCODING is the encoding used to save the wikipedia articles conten
        Default: 'utf-8'
    """
    print('Downloading files according to a key {} to directory {}'.format(titles, save_directory))
    dataset.download_files(titles, save_directory, encoding)
    print('Succesfully Downloaded the files')


if __name__ == '__main__':
    download_raw_data()
