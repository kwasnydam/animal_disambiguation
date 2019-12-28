"""Code for the web API. When run with default parameters it loads the saved models and uses them to perform pred

The API is documented using flask_resftul_swagger. The documentation for different endpoints and methods is available
at /api/spec.html or by getting the raw json by /api/spec.html.

Endpoints
    /predict
Methods
    GET

Response
    200 - predicition, when correct query
    400 - INVALID QUERY, when incorrect query (without word 'mouse')
"""

import click
import os

from src.data import dataset
from src.model import model
import api


@click.command()
@click.argument('data_model_params', default=dataset.DEFAULT_VECTORIZER_SETTINGS)
@click.argument('data_model_path', type=click.Path(writable=True, dir_okay=False),
                default=os.path.join(dataset.DEFAULT_DATA_MODEL_DIRECTORY, 'data_model.pickle'))
@click.argument('classificator_parameters', default=model.DEFAULT_CLASSIFIER_SETTINGS)
@click.argument('classificator_path', type=click.Path(writable=True, dir_okay=False),
                default=os.path.join(model.DEFAULT_MODEL_DIRECTORY, 'trained_model.pickle'))
def run(data_model_params, data_model_path, classificator_parameters, classificator_path):
    params = {
        'data_model_params': data_model_params,
        'data_model_path': data_model_path ,
        'classificator_parameters': classificator_parameters,
        'classificator_path': classificator_path
    }
    app = api.create_app(params)
    app.run()


if __name__ == '__main__':
    run()
