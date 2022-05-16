import numpy as np
from injectable import load_injection_container

from service.faceservice.recognition.networkstructure.EmbeddedLayerBuilder import EmbeddedLayerBuilder
from service.faceservice.recognition.networkstructure.SiameseModel import SiameseModel
from service.faceservice.recognition.networkstructure.SiameseModelBuilder import SiameseModelBuilder


class Classifier:

    def classify_images(self, face_list1, face_list2, threshold=0.9):
        # Getting the encodings for the passed faces
        tensor1 = encoder.predict(face_list1)
        tensor2 = encoder.predict(face_list2)

        distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
        print(distance)
        prediction = np.where(distance <= threshold, distance, 1)
        return prediction

    def classify_image_and_tensor(self, face_list1, tensor, threshold=0.9):
        # Getting the encodings for the passed faces
        tensor1 = encoder.predict(face_list1)

        distance = np.sum(np.square(tensor1 - tensor), axis=-1)
        print(distance)
        prediction = np.where(distance <= threshold, distance, 1)
        return prediction


siamese_network = SiameseModelBuilder().get_siamese_model()
siamese_network.summary()
siamese_model = SiameseModel(siamese_network)
siamese_model.load_weights("siamese_model-final")

embedded_layer_builder = EmbeddedLayerBuilder()
encoder = embedded_layer_builder.copy_weight_to_embedded_layer(siamese_model)
encoder.save_weights("encoder")
encoder.summary()
