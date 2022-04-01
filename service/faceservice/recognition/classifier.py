import time

import numpy as np

from service.faceservice.recognition.encoder import get_encoder
from service.faceservice.recognition.network_model import SiameseModel, get_siamese_network


class Classifier:

    def __init__(self):
        self.__weight = 128
        self.__height = 128
        self.__channels_amount = 3

    def get_weight(self):
        return self.__weight

    def get_height(self):
        return self.__height

    def extract_encoder(self, model):
        layer_encoder = get_encoder((self.__weight, self.__height, self.__channels_amount))
        i = 0
        for e_layer in model.layers[0].layers[3].layers:
            layer_weight = e_layer.get_weights()
            layer_encoder.layers[i].set_weights(layer_weight)
            i += 1
        return layer_encoder

    def classify_images(self, face_list1, face_list2, threshold=0.9):
        # Getting the encodings for the passed faces
        tensor1 = encoder.predict(face_list1)
        tensor2 = encoder.predict(face_list2)

        distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
        prediction = np.where(distance <= threshold, distance, 1)
        return prediction


siamese_network = get_siamese_network()
siamese_network.summary()
siamese_model = SiameseModel(siamese_network)
siamese_model.load_weights("siamese_model-final")

classifier = Classifier()
encoder = classifier.extract_encoder(siamese_model)
encoder.save_weights("encoder")
encoder.summary()
