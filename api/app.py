from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

from disambiguator.stub_model import DummyModel as Model

app = Flask(__name__)
api = Api(app)
model = Model()
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class CategorizeMouse(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        prediction = model.predict(user_query)

        output = {'prediction': prediction}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(CategorizeMouse, '/')

if __name__ == '__main__':
    app.run(debug=True)
