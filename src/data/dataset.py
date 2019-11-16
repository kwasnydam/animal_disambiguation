import os
import codecs

import wikipedia
from nltk.tokenize import sent_tokenize

DEFAULT_TITLES = {
    'animal': ('mouse', 'kangaroo mouse', 'hopping mouse'),
    'device': ('computer mouse', 'optical mouse')
}

up = os.path.dirname
DEFAULT_ROOT_DIRECTORY = up(up(up(__file__)))   # Get directory two levels above
DEFAULT_DATA_DIRECTORY = os.path.join(DEFAULT_ROOT_DIRECTORY, 'data')
DEFAULT_RAW_DATA_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, 'raw')
DEFAULT_INTERIM_DATA_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, 'interim')
DEFAULT_PROCESSED_DATA_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, 'processed')
DEFAULT_PROCESSED_TEXT_DATA_DIRECTORY = os.path.join(DEFAULT_PROCESSED_DATA_DIRECTORY, 'text')

DEFAULT_ENCODING = 'utf-8'


def download_files(titles, save_directory, encoding):
    for context in titles.keys():
        for title in titles[context]:
            save_filepath = os.path.join(save_directory, context, '{}.txt'.format(title))
            raw = try_download_file(title)
            with codecs.open(save_filepath, 'w', encoding) as ofile:
                ofile.write(raw)


def try_download_file(title):
    try:
        raw = wikipedia.page(title).content
    except wikipedia.exceptions.DisambiguationError as e:
        print('DISAMBIGUATION ERROR: Title: {}, Content: {}'.format(title, e))
        raise e
    except wikipedia.exceptions.PageError as e:
        print('PAGE ERROR: no wikipedia page matched query: {}'.format(title))
        raise e
    except wikipedia.exceptions.HTTPTimeoutError as e:
        print('TIMEOUT ERROR. Please check your connection and try again')
        raise e
    except Exception as e:
        print('UNKNOWN EXCEPTION WHEN DOWNLOADING {}'.format(title))
        raise e
    return raw


# define processing pipeline

def process_text(text):
    """Processes raw text downloaded from wikipedia

    Input:
        content of the wikipedia page or any other string

    Output:
        list of sentences containing word mouse
    """

    def split_into_lines(text):
        return text.splitlines()

    def split_into_sentences(lines):
        sentences = []
        for line in lines:
            sentences.extend(sent_tokenize(line))
        return sentences

    def filter_normalize_sentences(sentences):
        valid_sentences = []
        for sentence in sentences:
            if 'mice' in sentence:
                sentence = sentence.replace('mice', 'mouse')

            if 'mouse' in sentence:
                valid_sentences.append(sentence)
        return valid_sentences

    lines = split_into_lines(text)
    sentences = split_into_sentences(lines)
    valid_sentences = filter_normalize_sentences(sentences)
    return valid_sentences


def build_text_dataset(read_directory, save_directory, encoding):
    """Based on filtered  and normalized data in READ_DIRECTORY create single csv file

    Structure of the created file will be as follows:
    sentence;class

    Sentences from 'animal.txt' are class 'animal'
    Sentences from 'device.txt' are class 'device
    """
    save_dir = os.path.join(save_directory, 'dataset.csv')
    with codecs.open(save_dir, 'w', encoding) as of:
        for context in ['animal', 'device']:
            read_path = os.path.join(read_directory, '{}.txt'.format(context))
            with codecs.open(read_path, 'r', encoding) as rf:
                sentences = rf.read().splitlines()
                for sentence in sentences:
                    output_line = '{};{}\n'.format(sentence, context)
                    of.write(output_line)

