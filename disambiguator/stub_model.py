import numpy as np

class DummyModel:

    def __init__(self):
        self.answers = ['device', 'animal']

    def predict(self, query):
        random_index = np.random.randint(0, 1)
        return self.answers[random_index]
