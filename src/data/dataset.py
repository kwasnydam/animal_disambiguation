import os
import codecs
import string
import pickle
from collections import defaultdict

import wikipedia
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

DEFAULT_TITLES = {
    'animal': ('mouse', 'kangaroo mouse', 'hopping mouse'),
    'device': ('computer mouse', 'optical mouse')
}

DEFAULT_VECTORIZER_SETTINGS = {
    'ngram_range': (1, 3),
    'min_df': 2,
    'max_df': 0.5
}

up = os.path.dirname
DEFAULT_ROOT_DIRECTORY = up(up(up(__file__)))   # Get directory two levels above
DEFAULT_DATA_DIRECTORY = os.path.join(DEFAULT_ROOT_DIRECTORY, 'data')
DEFAULT_RAW_DATA_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, 'raw')
DEFAULT_INTERIM_DATA_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, 'interim')
DEFAULT_PROCESSED_DATA_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, 'processed')
DEFAULT_PROCESSED_TEXT_DATA_DIRECTORY = os.path.join(DEFAULT_PROCESSED_DATA_DIRECTORY, 'text')

DEFAULT_FEATURES_DIRECTORY = os.path.join(DEFAULT_PROCESSED_DATA_DIRECTORY, 'features')
DEFAULT_DATA_MODEL_DIRECTORY = os.path.join(DEFAULT_ROOT_DIRECTORY, 'models')

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

    punctuation = set(string.punctuation)

    save_dir = os.path.join(save_directory, 'dataset.csv')
    with codecs.open(save_dir, 'w', encoding) as of:
        for context in ['animal', 'device']:
            read_path = os.path.join(read_directory, '{}.txt'.format(context))
            with codecs.open(read_path, 'r', encoding) as rf:
                sentences = rf.read().splitlines()
                for sentence in sentences:
                    # removing punctuation as it carries little info yet could break our csv file
                    s = ''.join(ch for ch in sentence if ch not in punctuation)
                    output_line = '{};{}\n'.format(s, context)
                    of.write(output_line)

class TextLabelsVectorizer:
    """Responsible for vectorization of text into feature vector (tfidf) and classes labels into binary labels

    The interface follows sklearn conventions, so it has following methods:
    fit(text_corpus, class_labels) - fits the vectorizer to the given text corpus
    transform(text, class_label=None) - using fitted parameters transforms the text and return it
    fit_transform(text_corpus, class_labels) - peform fit(...) then transform
    get_params() - returns parameters of a fitted Vectorizer
    set_params(params) - load the vectorizer with given set of prefit params
    Attributes:
        label_encoder - object transforming categorical labels into numerical values (0,1)
        vectorizer - object responsible for vectorization of text with tfidf method
    """

    def __init__(self, vectorizer_params):
        self.label_encoder = MoreFrequentZeroLabelEncoder()
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self.tokenize,
            **vectorizer_params
        )
        self.save_name = self._get_save_name(vectorizer_params)
        self.stopwords = stopwords.words('english')

    def tokenize(self, text):
        """Performs tokenization of input sentence

        First, using nltk word_tokenize splits the sentence into tokens
        Then, lowercases all tokens
        Finally, removes stopwords tokens and digits
        return a list of valid tokens

        Args:
            text: string representing a single sentence.

        Returns:
            List of tokens
        """
        words = word_tokenize(text)
        words = [w.lower() for w in words]
        return [w for w in words if w not in self.stopwords and not w.isdigit()]

    def fit(self, text_corpora, class_labels):
        """Fits the estimator to the data.

        For text, performs tokenization with punctuation, stopwords and digits removal, followed bytfidf vectorization.
        For labels, perform label encoding, storing the labels corresponding to numerical classes

        Args:
            text_corpora - iterable of strings containing sentences to fit the tfidf on
            class_labels - iterbale of class labels to fit into numerical labels

        Returns:
            None
        """
        self.vectorizer.fit(text_corpora)  # train on variables from train set (dont pass classes)
        self.label_encoder.fit(class_labels)

    def transform(self, text, class_labels=None):
        """Transforms input text into feature vector using fit parameters. If class_label not None, encodes it.

        Args:
            text - string or list of strings conataining sentences to vectorize
            class_labels (optional) - class label or list of class labels to encode

        Returns:
            vectorized_text - numpy array of shape(num_sentences, num_fit_features)
            encoded_labels - if class_label provided, return encoded class label
        """
        features = self.vectorizer.transform(text.copy())
        features = features.todense()   # we dont want sparse matirces onjects
        if class_labels is not None:
            encoded_labels = self.label_encoder.transform(class_labels)
            return features, encoded_labels
        else:
            return features

    def fit_transform(self, text, class_labels):
        """Performs fit, then transform. Look fit and transform documentation"""
        self.fit(text, class_labels)
        return self.transform(text, class_labels)

    def get_params(self):
        """Returns mapping of {Attirbutes: attribute_params}"""
        params = {
            'label_encoder': self.label_encoder.get_params(),
            'vectorizer': self.vectorizer.get_params()
        }
        return params

    def set_params(self, params):
        """Sets parameters of label encoder and vectorizer to params.

        Args:
            params - mapping {attribute: atrribute_parameters} gotten from TextLabelsVectorizer.get_params()
        """
        self.label_encoder.set_params(params['label_encoder'])
        self.vectorizer.set_params(**params['vectorizer'])

    def get_classes_name(self, binary_classes):
        return self.label_encoder.inverse_transform(binary_classes)

    def _get_save_name(self, params):
        maxdf = params['max_df']
        mindf = params['min_df']
        ngrams = '{}{}'.format(*params['ngram_range'])

        return 'maxdf_{}_mindf_{}_ngrams_{}'.format(maxdf, mindf, ngrams)


class MoreFrequentZeroLabelEncoder:

    def __init__(self):
        self.labels = {}
        self.inverse_mapping = {}

    def fit(self, classes):
        available_labels, counts = np.unique(classes, return_counts=True)
        labels_counts = zip(available_labels, counts)
        labels_counts = sorted(labels_counts, key=lambda x: x[1], reverse=True)
        print(labels_counts)
        for index, label in enumerate(labels_counts):
            self.labels[label[0]] = index
            self.inverse_mapping[str(index)] = label[0]

    def transform(self, classes):
        result = np.zeros(classes.shape).astype(np.int32)
        for label in self.labels.keys():
            result[classes == label] = self.labels[label]
        return result

    def inverse_transform(self, encoded_class):
        output = []
        try:
            for idx in range(encoded_class.size):
                el = encoded_class[idx]
                output.append(self.inverse_mapping[str(el)])
        except Exception as e:
            output = self.inverse_mapping[str(encoded_class)]

        return output

    def get_params(self):
        params = {
            'labels': self.labels,
            'inverse_mapping': self.inverse_mapping
        }
        return params

    def set_params(self, params):
        self.labels = params['labels']
        self.inverse_mapping = params['inverse_mapping']


def build_dataset_and_datamodel(read_directory, data_save_directory, data_model_save_directory,vectorizer_params):
    text_vectorizer = TextLabelsVectorizer(vectorizer_params)
    train_filepath = os.path.join(read_directory, 'train.csv')
    validation_filepath = os.path.join(read_directory, 'validation.csv')
    train_data = pd.read_csv(train_filepath, sep=';')
    validation_data = pd.read_csv(validation_filepath, sep=';')

    # fit to train dataset
    text_vectorizer.fit(train_data.iloc[:,0], train_data.iloc[:,1])
    extract_and_save_features(
        train_data.iloc[:,0],
        train_data.iloc[:,1],
        os.path.join(data_save_directory, 'train.npy'),
        text_vectorizer.transform
                              )
    print('Train features avaiable at {}'.format(data_save_directory))

    extract_and_save_features(
        validation_data.iloc[:, 0],
        validation_data.iloc[:, 1],
        os.path.join(data_save_directory, 'validation.npy'),
        text_vectorizer.transform
    )
    print('Validation features avaiable at {}'.format(data_save_directory))

    # save data model (will be used on evaluation and in production)
    with open(os.path.join(DEFAULT_DATA_MODEL_DIRECTORY, 'data_model.pickle'), 'wb') as of:
        pickle.dump(text_vectorizer.get_params(), of, protocol=pickle.HIGHEST_PROTOCOL)
    print('Data model avaiable at {}'.format(data_model_save_directory))


def extract_and_save_features(data, classes, save_path, extract_function):
    features, classes = extract_function(data, classes)
    dataset = np.hstack((features, classes.reshape(classes.size, -1)))
    np.save(os.path.join(save_path), dataset)

