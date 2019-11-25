"""Runs the flask web application with a single endpoint, which accepts queries and outputs prediction"""

import pickle
import os

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

import src.model.model as model
import src.data.dataset as dataset


def try_opening_file_pickle(path):
    try:
        with open(path, 'rb') as f:
            file_content = pickle.load(f)
    except FileNotFoundError as e:
        print('FileNotFound exception occured when trying to open: {}. Disambiguator build failed.'.format(
            path
        ))
        raise e
    except Exception as e:
        print('Exception occured when trying to open {}: {}'.format(path, e))
        raise e

    return file_content


app = Flask(__name__)
api = Api(app)
disambiguator = try_opening_file_pickle(os.path.join(model.DEFAULT_MODEL_DIRECTORY, 'trained_model.pickle'))
data_model = try_opening_file_pickle(os.path.join(dataset.DEFAULT_DATA_MODEL_DIRECTORY, 'data_model.pickle'))
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class CategorizeMouse(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        features = data_model.transform(user_query)
        prediction, prediction_prob = disambiguator.predict(features, format='text')[:,
                                             0], disambiguator.predict(features)[:, 1]
        output = {'prediction': prediction[0]}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(CategorizeMouse, '/')

if __name__ == '__main__':
    app.run(debug=True)
