# Author

Damian Kwasny

# Overview

The purpose of the project is to create a web service with a simple API and a
single endpoint, which would accept an input sentence containing the word
_mouse_ and return whether this is an animal or a peripheral.

# Requirements

* it must be implemented as a web service
* it must accept an input sentence and return a classification result

# Milestones

| Milestone | Completed Hours | Estimated Cost | Estimated Finish | Actual Finish |
|:---------:|:---------------:|:--------------:|:----------------:|---------------|
|Propose architecture and estimate work|2|6                |08.11                  |               |
|   |   |   |   |   |


# Technical Architecture

There are at least 3 separate tasks that needs to be accounted for:

   1. The API, handling the requests and generating a response.
   2. The model, trained on the data and ready for evaluation.
   3. The data to train and crossvalidate the model on.

## Proposed Solution

Each of the parts is equally important to make the system work. Without the
user interface, the application cannot communicate with the user and is useless.
Without the data it is impossible to train a valid model that provides a valid
output.

The following steps needs to be undertaken:


  * Since for now I have no idea about how to run a webservice, I need to dig into tutorials and preferably create a dummy API that would accept a
request and return some random output that fits the requirements
  * I need to research about a kind of ML models that could be used in this task. For the time being some simple bag of words representation and building a dictionary seems like a plausible option.
  * I need to collect some data on which I could train my classifier. For the computer mouse context, some PC-related articles and tutorials seems like a good start. For the animal part it is kind of more tricky, some animal dictionaries and stuff comes to mind. I need to prototype on some small, hand picked prototpyes and decide whether more data (web scrapping?) is needed
