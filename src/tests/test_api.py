import unittest
import os

from api import create_app
from src.data import dataset
from src.model import model


class TestApp(unittest.TestCase):
    def config_testing(self):
        params = {
            'data_model_params': dataset.DEFAULT_VECTORIZER_SETTINGS,
            'data_model_path': os.path.join(dataset.DEFAULT_DATA_MODEL_DIRECTORY, 'data_model.pickle'),
            'classificator_parameters': model.DEFAULT_CLASSIFIER_SETTINGS,
            'classificator_path': os.path.join(model.DEFAULT_MODEL_DIRECTORY, 'trained_model.pickle')
        }
        app = create_app(params)
        app.config.from_mapping({'TESTING': True})
        client = app.test_client()
        return client

    def test_given_sentence_containing_mouse_return_valid_answer_with_code_200(self):
        sentences = ['mouse is a rodent', 'move the mouse pointer']
        test_client = self.config_testing()
        for sentence in sentences:
            response = test_client.get('/predict?query={}'.format(sentence))
            self.assertTrue(response.json['prediction'] in ('animal', 'device'))
            self.assertTrue(response.status == '200 OK')

    def test_given_sentence_not_containing_mouse_return_code_404(self):
        sentences = ['xyz is a rodent', 'move the xyz pointer']
        test_client = self.config_testing()
        for sentence in sentences:
            response = test_client.get('/predict?query={}'.format(sentence))
            self.assertTrue(response.status == '404 NOT FOUND')


if __name__ == '__main__':
    unittest.main()
