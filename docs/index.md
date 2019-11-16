# Author

Damian Kwasny

# Overview

The purpose of the project is to create a web service with a simple API and a
single endpoint, which would accept an input sentence containing the word
_mouse_ and return whether it is an animal or a peripheral.

# Requirements

* it must be implemented as a web service
* it must accept an input sentence and return a classification result

# Milestones

| Milestone | Completed Hours | Estimated Cost | Estimated Finish | Actual Finish |
|:---------:|:---------------:|:--------------:|:----------------:|---------------|
|Propose architecture and estimate work|4|6                |08.11                  |        08.11       |
|Implement a simple web service skeleton using Flask RESTful API   | 4  | 6  | 10.11  |  |
|Think out the way to obtain a reliable training and testing data   | 0  | 8  | 13.11  |   |
|Collect and preprocess the data   | 0  | 6  | 14.11  |   |
|Research on the model viable for the disambiguation task   | 0  | 6  | 15.11  |   |
|Implement and Validate the model   | 0  | 6  | 16.11  |   |
|Integrate and test the application   |  0 | 6  | 17.11  |   |
|Extend the design and functionality of the API   | 0  |  3 | 17.11  |   |
|Bug fixes and deployment|   | 12  | 19.11  |   |


# Technical Architecture

There are at least 3 separate tasks that needs to be accounted for:

   1. The API, handling the requests and generating a response.
   2. The model, trained on the data and ready for evaluation.
   3. The data to train and crossvalidate the model on.

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

Tech stack:
Python, 3rd party libraries: Flask, scikit-learn, numpy, pandas

Why these?
  * Python - a go-to programming language for ML and fast prototyping
  All of the used 3rd party libraries are released under a permissive BSD License and thus can be used commercially with a simple mention.
  * Flask - A lightweighted framework for web service creation
  * scikit-learn - a go-to ML framework containing a lot of different models, perfect for creating a prototype solution
  * numpy - essential all there where computation are involved
  * pandas - a handy library to handle the data processing


Since the web API part was initially the most mysterious I have done some search on that and decided, that
I will design the application as a Flask RESTful API app. It will accept a user query and pass it down the model as well as send the results obtained from the model. The decision is motivated by the fact, that Flask seems like a lightweighted and easy to set up choice for building a working prototype and that is
ultimately the most important part when starting on with a new ML project.

Data collection method and model choice are still WIP.

The application needs to be structured so that the API depends only on a simple interface (like model.predict(sentence)) and multiple models can be used.

## Testing
More once the data piepline will be established
