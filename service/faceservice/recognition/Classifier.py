import numpy as np

from service.faceservice.recognition.networkstructure.EmbeddedLayerBuilder import EmbeddedLayerBuilder
from service.faceservice.recognition.networkstructure.SiameseModel import SiameseModel
from service.faceservice.recognition.networkstructure.SiameseModelBuilder import SiameseModelBuilder


class Classifier:

    def classify_images(self, face1, face_list2, threshold=0.9):
        # Getting the encodings for the passed faces
        tensor1 = encoder_model.predict(face1)
        tensor2 = encoder_model.predict(face_list2)

        distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
        print(distance)
        prediction = np.where(distance <= threshold, distance, 1)
        return prediction


siamese_network = SiameseModelBuilder().get_siamese_model()

siamese_model = SiameseModel(siamese_network)
siamese_model.load_weights("siamese_model-final")

embedded_layer_builder = EmbeddedLayerBuilder()
encoder_model = embedded_layer_builder.copy_weight_to_encode_layer(siamese_model)
encoder_model.summary()
