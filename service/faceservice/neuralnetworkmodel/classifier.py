import time

import numpy as np

from service.faceservice.neuralnetworkmodel.encoder import get_encoder
from service.faceservice.neuralnetworkmodel.networkmodel import SiameseModel, get_siamese_network


def extract_encoder(model):
    layer_encoder = get_encoder((128, 128, 3))
    i = 0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        layer_encoder.layers[i].set_weights(layer_weight)
        i += 1
    return layer_encoder


def classify_images(face_list1, face_list2, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    start_time = time.time()
    tensor2 = encoder.predict(face_list2)
    print("--- %s seconds ---" % (time.time() - start_time))

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


siamese_network = get_siamese_network()
siamese_network.summary()
siamese_model = SiameseModel(siamese_network)
siamese_model.load_weights("siamese_model-final")
encoder = extract_encoder(siamese_model)
encoder.save_weights("encoder")
encoder.summary()
