"""Code for the web API. When run with default parameters it loads the saved models and uses them to perform pred

The API is documented using flask_resftul_swagger. The documentation for different endpoints and methods is available
at /api/spec.html or by getting the raw json by /api/spec.html.

Endpoints
    /predict
Methods
    GET

Response
    200 - predicition, when correct query
    404 - INVALID QUERY, when incorrect query (without word 'mouse')
"""

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_restful_swagger import swagger

from src.model import model


def create_app(model_params, app_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app = Flask(__name__)
    api = swagger.docs(Api(app), apiVersion='0.1')
    # build model
    disambiguator = model.build_mmdisambiguator(**model_params)
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
            self._abort_if_invalid_query(user_query)

            prediction_results = disambiguator.predict(user_query, format='text', source='text')
            prediction, prediction_probability = prediction_results[:, 0], prediction_results[:, 1]
            output = {'prediction': prediction[0]}

            return output

        def _abort_if_invalid_query(self, query):
            """Abort if the user query does not contain word 'mouse'"""
            if 'mouse' not in query.lower():
                abort(404, message="INVALID QUERY: Query {} does not contain word 'mouse'".format(query))

    # Setup the Api resource routing here
    # Route the URL to the resource
    api.add_resource(CategorizeMouse, '/predict')
    return app
