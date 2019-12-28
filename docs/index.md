# Author

Damian Kwasny

# Overview

The purpose of the project is to create a web service with a simple API and a
single endpoint, which would accept an input sentence containing the word
_mouse_ and return whether it is an animal or a peripheral.

# Requirements

* it must be implemented as a web service
* it must accept an input sentence and return a classification result

# Technical Architecture

There are at least 3 separate tasks that needs to be accounted for:

   1. The API, handling the requests and generating a response.
   2. The model, trained on the data and ready for evaluation.
   3. The data to train and validate the model on.

## Problem Breakdown

Each of the parts is equally important to make the system work. Without the
user interface, the application cannot communicate with the user and is useless.
Without the data it is impossible to train a valid model that provides a valid
output.

The following steps needs to be undertaken:

  * Since for now I have no idea about how to run a webservice, I need to dig into tutorials and preferably create a dummy API that would accept a
request and return some random output that fits the requirements
  * I need to research about a kind of ML models that could be used in this task. For the time being some simple bag of words representation and building a dictionary seems like a plausible option.
  * I need to collect some data on which I could train my classifier. For the computer mouse context, some PC-related articles and tutorials seems like a good start. For the animal part it is kind of more tricky, some animal dictionaries and stuff comes to mind. I need to prototype on some small, hand picked prototpyes and decide whether more data (web scrapping?) is needed

## Proposed Solution:

### Tech stack:
Git, Python, 3rd party libraries: Flask, scikit-learn, numpy, pandas, nltk, jupyter notebook

Why these?
  * Python - a go-to programming language for ML and fast prototyping
  All of the used 3rd party libraries are released under a permissive BSD License and thus can be used commercially with a simple mention.
  * Flask - A lightweighted framework for web service creation
  * scikit-learn - a go-to ML framework containing a lot of different models, perfect for creating a prototype solution
  * numpy - essential all there where computation are involved
  * pandas - a handy library to handle the data processing
  * nltk - BSD licensed toolkit for fast prototyping of NLP solutions, perfect for text manipulation pipeline and simple models
  * jupyter notebook - great, bsd licensed tool for fast prototyping and experiment documentation

### Application Architecture
 *  API - A simple RESTful API built with the Flask framework. Flask is a lightweighted choice good for prototyping.1
 The API accepts the GET query, check if it is valid (contains word mouse) and call the model to make the prediciton.
 It will then return the response containing the model prediction. There is no extensive data processing needed on the part of the API, as it
 is part of the model pipeline. 

 *  Model - I have decided to build a model around the following concepts:
    *   Task - Binary Classification
    *   data - A simple LabelEncoder that maps more common class in the datset into 0 and a TfIdf vectorizer that
    performs the text transformation onto numerical features (TfIdf features)
    *   model - Logistic Regression. It is simple, provides a probability output, so the decision can be postponed and
    based on the task at hand while manipulating the threshold. Also, it's weights are interpretable.

 *  Data - I have built my own dataset around the wikipedia articles:
    *   With 'wikipedia' module I have built my own, small dataset consisting of some articles about mouse in device
    and animal context. The dataset is skewed towards the 'device' class, however with this method it is easy to extend
    the datset by providing more articles in the datset download script and rebuilding it. The code for data aquisition is
    part of the repository.

The application needs to be structured so that the API depends only on a simple interface (like model.predict(sentence)) and multiple models can be used.

## Testing

Unit Testing:
  * Simple checks if the data transfromations and models given some well established inputs are doing their job
  * Edge cases (empty queries, queires without the key word, ambigous queries (ex. 'it is mouse'))

Correctness:
  * Dummy classifier: predicts the majority class from the data set all the estimate
  * The model we develop must beat the dummy classifier (must be skillful)
  * Performance measures:
    * accuracy
    * precision
    * recall
    * f1 score
    * confussion matrix

  With further experiments we could construct the Precision/Recall curves (suitable for imbalanced dataset as the one
  I have used in my experiments)
