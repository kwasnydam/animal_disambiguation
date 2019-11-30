"""Runs the flask web application with a single endpoint, which accepts queries and outputs prediction"""

import pickle
import os

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_restful_swagger import swagger

import src.model.model as model
import src.data.dataset as dataset


def abort_if_invalid_query(query):
    """Abort if the user query does not contain word 'mouse'"""
    if 'mouse' not in query.lower():
        abort(400, message="INVALID QUERY: Query {} does not contain word 'mouse'".format(query))


app = Flask(__name__)
api = swagger.docs(Api(app), apiVersion='0.1')
# build model
disambiguator = model.build_mmdisambiguator(
    data_model_params=dataset.DEFAULT_VECTORIZER_SETTINGS,
    data_model_path=os.path.join(dataset.DEFAULT_DATA_MODEL_DIRECTORY, 'data_model.pickle'),
    classificator_parameters=model.DEFAULT_CLASSIFIER_SETTINGS,
    classificator_path=os.path.join(model.DEFAULT_MODEL_DIRECTORY, 'trained_model.pickle')
)
parser = reqparse.RequestParser()
parser.add_argument('query')


class CategorizeMouse(Resource):
    "Find user query, determine if it is valid and tell model to make predicition. Return response with prediction"
    @swagger.operation(
        responseClass='GET',
        nickname='get',
        parameters=[
            {
                "name": "query",
                "description": "Sentence containing word mouse to perform prediction upon.",
                "required": True,
                "allowMultiple": False,
                "dataType": 'string',
                'paramType': 'query'
            }
        ]
    )
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        abort_if_invalid_query(user_query)

        prediction_results = disambiguator.predict(user_query, format='text', source='text')
        prediction, prediction_probability = prediction_results[:,0], prediction_results[:,1]
        output = {'prediction': prediction[0]}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(CategorizeMouse, '/')

if __name__ == '__main__':
    app.run(debug=True)
