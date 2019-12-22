# Description

A simple project, which aim is to create a simple web service that accepts an
input sentence containing the word _mouse_ and returns whether the _mouse_ is
an animal or a peripherial.

# Installation

For a quick start, make sure you have GNU _make_ installed. For windows, you can do it by using an extension for git bash
First, run `make .venv`.
Next, activate the virtual environment. (OS dependent operation)
Now, you can finish the setup with `make requirements`

# Usage
In order to run the service, type `make run`. Once it loads, you can send requests to the API located at localhost:5000/predict.
The API includes a single endpoint @/predict and a GET method. You need to supply the query key with the value being
the sentence you want to perform the prediction on. The sentence must contain a word _mouse_.

# Tests & Development
To run tests, please use the command `make tests`.
To develop the model and reproduce the results, please run `make data`. It will perform the whole default data processing
pipeline. Check the _Makefile_ for details of operations being performed.

# License
The project is released under the _MIT_ license (see LICENSE)
