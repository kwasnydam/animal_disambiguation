"""From RAW data filter out sentences not containing the word mouse or mice."""
import click
import codecs
import os

import src.data.dataset as dataset


@click.command()
@click.argument('read_directory', type=click.Path(dir_okay=True),
                default=dataset.DEFAULT_RAW_DATA_DIRECTORY)
@click.argument('save_directory', type=click.Path(writable=True, dir_okay=True),
                default=dataset.DEFAULT_INTERIM_DATA_DIRECTORY)
@click.argument('encoding', default=dataset.DEFAULT_ENCODING)
def filter_raw_data(read_directory, save_directory, encoding='utf-8'):
    """Filter out sentences not containing 'mouse' and change 'mice' to 'mouse'

    READ_DIRECTORY is directory to read raw data from.
        Default: <project_root>/data/raw
    SAVE_DIRECTORY is directory to store filtered  data.
        Default: <project_root>/data/interim
    ENCODING is the encoding used to save the filtered sentences
        Default: 'utf-8'

    Creates files 'animal.txt' and 'device.txt' in SAVE_DIRECTORY
    """
    for context in ['animal', 'device']:
        read_dir = os.path.join(read_directory, context)
        save_dir = os.path.join(save_directory, '{}.txt'.format(context))
        filenames = [filename for filename in os.listdir(read_dir) if filename.endswith('.txt')]
        print(filenames)

        with codecs.open(save_dir, 'w', encoding) as of:
            for filename in filenames:
                read_path = os.path.join(read_dir, filename)
                with codecs.open(read_path, 'r', encoding) as rf:
                    text = rf.read()
                    processed_text = dataset.process_text(text)
                    for sentence in processed_text:
                        of.write(sentence)
                        of.write('\n')


if __name__ == '__main__':
    filter_raw_data()
