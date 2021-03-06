"""dataset module contains classes and helper functions used to perform various datset related actions.

Helper functions:
    download_files - download files from wikipedia
    try_download_file -  try downloading a single article from wikipedia
    process_text - process raw text from wikiepdia into valid sentences that can be used to build the dataset
    build_text_dataset - build dataset from valid sentences
    build_dataset_and_datamodel - from text dataset, build features and save them as features dataset, save the vectorizer

Classes:
    TextLabelsVectorizer - resposnible for transforming sentences into numerical representatio
    MoreFrequentZeroLabelEncoder - Label Encoder which assign 0 to more frequent class ina training dataset
"""

import os
import codecs
import string
import pickle

import wikipedia
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


DEFAULT_TITLES = {
    'animal': ('mouse', 'kangaroo mouse', 'hopping mouse'),
    'device': ('computer mouse', 'optical mouse')
}

DEFAULT_VECTORIZER_SETTINGS = {
    'ngram_range': (1, 3),
    'min_df': 2,
    'max_df': 1.0
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
    """Based on titles, download articles and save to save_directory with encoding.

    Args
        titles - dictionary of form {context: wikiepdia_article_title}
        save_directory - to save downlaoded data
        encoding - to use to save the data
    """
    for context in titles.keys():
        for title in titles[context]:
            save_filepath = os.path.join(save_directory, context, '{}.txt'.format(title))
            raw = try_download_file(title)
            with codecs.open(save_filepath, 'w', encoding) as ofile:
                ofile.write(raw)


def try_download_file(title):
    """Try downloading the article with given title from wikipedia."""
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


def train_validation_test_split(read_path, save_directory, encoding):
    """Based on the processed text files from read_path create train-validation-test split"""
    text_df = pd.read_csv(
        read_path, sep=';', header=None, names=['text', 'class']
        )
    # let's split our data into 3 sets: train, validation, test with a fixed random seed for reproducibility
    random_seed = 0
    np.random.seed(0)
    train_prop = 0.6
    valid_prop = 0.2
    train, validate, test = np.split(
        text_df.sample(frac=1),
        [int(train_prop*len(text_df)), int((train_prop+valid_prop)*len(text_df))]
    )

    # Finally, lets save our data
    filenames = ['train', 'validation','test']
    filenames = ['{}.csv'.format(filename) for filename in filenames]
    data_splits = [train, validate, test]
    for data, filename in zip(data_splits, filenames):
        save_path = os.path.join(save_directory, filename)
        data.to_csv(save_path, sep=';', index=False)

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
        self.vocabulary = None

    def tokenize(self, text):
        """Performs tokenization of input sentence

        First, using nltk word_tokenize splits the sentence into tokens
        Then, lowercases all tokens and stem the tokens
        Finally, removes stopwords tokens and digits
        return a list of valid tokens

        Args:
            text: string representing a single sentence.

        Returns:
            List of tokens
        """
        stemmer = PorterStemmer()
        punctuation = set(string.punctuation)
        words = word_tokenize(text)
        words = [w.lower() for w in words]
        words = [stemmer.stem(word) for word in words]
        tokens = [w for w in words if w not in self.stopwords and not w.isdigit() and w not in punctuation]
        return tokens

    def is_fitted(self):
        """Check if the classifier is fitted."""
        try:
            check_is_fitted(self.vectorizer, '_tfidf')
        except NotFittedError:
            return False
        return True

    def _is_valid_data(self, data):
        if isinstance(data, list):
            return len(data) > 0
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return not data.empty
        else:
            raise TypeError('Unsupported data type')

    def fit(self, text_corpora, class_labels):
        """Fits the estimator to the data.

        For text, performs tokenization with punctuation, stopwords and digits removal, followed bytfidf vectorization.
        For labels, perform label encoding, storing the labels corresponding to numerical classes

        Args:
            text_corpora - pd.Series or list of strings containing sentences to fit the tfidf on
            class_labels - pd.Series or list of class labels to fit into numerical labels

        Returns:
            None
        """
        if not self._is_valid_data(text_corpora):
            raise ValueError('Empty sentences')
        if not self._is_valid_data(class_labels):
            raise ValueError('Empty labels')

        self.vectorizer.fit(text_corpora)  # train on variables from train set (dont pass classes)
        self.fit_labels(class_labels)

    def fit_labels(self, class_labels):
        if not self._is_valid_data(class_labels):
            raise ValueError('Empty labels')
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
        try:
            # we want a list not a string
            assert type(text) is not str
        except AssertionError as e:
            text = [text]

        try:
            features = self.vectorizer.transform(text.copy())
        except NotFittedError:
            raise NotFittedError('The TfIdf vectorizer has not been fit. Fit the Vectorizer before attemmpting '
                                 'to transform')

        features = features.todense()   # we dont want sparse matrices objects
        if class_labels is not None:
            encoded_labels = self.transform_labels(class_labels)
            return features, encoded_labels
        else:
            return features

    def transform_labels(self, labels):
        return self.label_encoder.transform(labels)

    def fit_transform(self, text, class_labels):
        """Performs fit, then transform. Look fit and transform documentation"""
        self.fit(text, class_labels)
        return self.transform(text, class_labels)

    def serialize(self):
        """Returns objects and parameters necessary to perform transformation"""
        to_serialize = {
            'label_encoder': self.label_encoder.get_params(),
            'vectorizer': self.vectorizer
        }
        return to_serialize

    def deserialize(self, serialized_representation):
        """Loads the serialized representation back into our object

        Args:
            serialized_representation - mapping {attribute: objects} gotten from TextLabelsVectorizer.serialize()
        """
        self.label_encoder.set_params(serialized_representation['label_encoder'])
        self.vectorizer = serialized_representation['vectorizer']

    def get_classes_name(self, binary_classes):
        return self.label_encoder.inverse_transform(binary_classes)

    def _get_save_name(self, params):
        maxdf = params['max_df']
        mindf = params['min_df']
        ngrams = '{}{}'.format(*params['ngram_range'])

        return 'maxdf_{}_mindf_{}_ngrams_{}'.format(maxdf, mindf, ngrams)


class MoreFrequentZeroLabelEncoder:
    """Label Encoder that treats more frequent class as 0 and is easily serialized

    Methods:
        fit - fit labels and inverse labels dictionaries to training data
        transform - using trained dictionaries encode the text labels into numerical classes
        inverse_transform - using trained dictionaries decode numerical classes into text labels
    """

    def __init__(self):
        self.labels = {}
        self.inverse_mapping = {}

    def fit(self, classes):
        """learn the mapping text labels -> numerical classes"""
        available_labels, counts = np.unique(classes, return_counts=True)
        labels_counts = zip(available_labels, counts)
        labels_counts = sorted(labels_counts, key=lambda x: x[1], reverse=True)
        for index, label in enumerate(labels_counts):
            self.labels[label[0]] = index
            self.inverse_mapping[str(index)] = label[0]

    def transform(self, classes):
        """map labels -> numerical classes"""
        try:
            result = np.zeros(classes.shape).astype(np.int32)
        except AttributeError:
            classes = np.asarray(classes)
            result = np.zeros(classes.shape).astype(np.int32)

        for label in self.labels.keys():
            result[classes == label] = self.labels[label]
        return result

    def inverse_transform(self, encoded_class):
        """map numercial classes -> labels"""
        output = []
        try:
            for idx in range(self._get_size(encoded_class)):
                el = encoded_class[idx]
                output.append(self.inverse_mapping[str(el)])
            output = np.asarray(output, dtype=str)
        except TypeError as e:
            output = self.inverse_mapping[str(encoded_class)]

        return output

    def _get_size(self, data):
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, pd.Series) or isinstance(data, np.ndarray):
            return data.size
        else:
            raise TypeError('Unsupported data of type {}'.format(type(data)))

    def get_params(self):
        params = {
            'labels': self.labels,
            'inverse_mapping': self.inverse_mapping
        }
        return params

    def set_params(self, params):
        self.labels = params['labels']
        self.inverse_mapping = params['inverse_mapping']


def build_dataset_and_datamodel(read_directory, data_save_directory, data_model_save_directory, vectorizer_params):
    """Based on the parameters and saving directories, build dataset and save vectorizer used to generate it.

    Builds both training and evaluation features
    Args
        read_directory - directory containing processed text data
        data_save_directory - directory to save features in
        data_model_save_directory - directory to save the fit vectorizer in
        vectorizer_params - parameters of the TfIdf vectorizer
    """
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
        serialized_data_model = text_vectorizer.serialize()
        pickle.dump(serialized_data_model, of)
    print('Data model avaiable at {}'.format(data_model_save_directory))


def extract_and_save_features(data, classes, save_path, extract_function):
    features, classes = extract_function(data, classes)
    dataset = np.hstack((features, classes.reshape(classes.size, -1)))
    np.save(os.path.join(save_path), dataset)

