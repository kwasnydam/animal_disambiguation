"""Code for the web API. When run with default parameters it loads the saved models and uses them to perform pred

The API is documented using flask_resftul_swagger. The documentation for different endpoints and methods is available
at /api/spec.html or by getting the raw json by /api/spec.html.

Endpoints
    /
Methods
    GET

Response
    200 - predicition, when correct query
    400 - INVALID QUERY, when incorrect query (without word 'mouse')
"""

import click
import os

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_restful_swagger import swagger

from src.data import dataset
from src.model import model


@click.command()
@click.argument('data_model_params', default=dataset.DEFAULT_VECTORIZER_SETTINGS)
@click.argument('data_model_path', type=click.Path(writable=True, dir_okay=False),
                default=os.path.join(dataset.DEFAULT_DATA_MODEL_DIRECTORY, 'data_model.pickle'))
@click.argument('classificator_parameters', default=model.DEFAULT_CLASSIFIER_SETTINGS)
@click.argument('classificator_path', type=click.Path(writable=True, dir_okay=False),
                default=os.path.join(model.DEFAULT_MODEL_DIRECTORY, 'trained_model.pickle'))
def run(data_model_params, data_model_path, classificator_parameters, classificator_path):
    def abort_if_invalid_query(query):
        """Abort if the user query does not contain word 'mouse'"""
        if 'mouse' not in query.lower():
            abort(404, message="INVALID QUERY: Query {} does not contain word 'mouse'".format(query))

    app = Flask(__name__)
    api = swagger.docs(Api(app), apiVersion='0.1')
    # build model
    disambiguator = model.build_mmdisambiguator(
        data_model_params=data_model_params,
        data_model_path=data_model_path,
        classificator_parameters=classificator_parameters,
        classificator_path=classificator_path
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
            prediction, prediction_probability = prediction_results[:, 0], prediction_results[:, 1]
            output = {'prediction': prediction[0]}

            return output

    # Setup the Api resource routing here
    # Route the URL to the resource
    api.add_resource(CategorizeMouse, '/')
    app.run()

if __name__ == '__main__':
    run()