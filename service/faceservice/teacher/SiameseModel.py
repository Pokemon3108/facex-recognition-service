import numpy as np
from keras import Input, Model, layers
from keras.layers import Dense
import tensorflow as tf

from teacher import EmbeddingLayerBuilder
from teacher.DistanceLayer import DistanceLayer


class SiameseModel:

    def __init__(self):
        self.__embedding_layer_builder = EmbeddingLayerBuilder()

    def get_siamese_model(self, input_shape):
        # Anchor image input in the network

        embedded_layer = self.__embedding_layer_builder.build_layer(input_shape)

        anchor_input = layers.Input(input_shape, name="Anchor_Input")
        positive_input = layers.Input(input_shape, name="Positive_Input")
        negative_input = layers.Input(input_shape, name="Negative_Input")

        # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
        distances = DistanceLayer()(
            embedded_layer(anchor_input),
            embedded_layer(positive_input),
            embedded_layer(negative_input)
        )

        # Creating the Model
        siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=distances,
            name="Siamese_Network"
        )
        return siamese_network
