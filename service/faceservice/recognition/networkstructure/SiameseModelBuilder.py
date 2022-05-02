from keras import layers
from keras.models import Model

from service.faceservice.recognition.networkstructure.DistanceLayer import DistanceLayer
from service.faceservice.recognition.networkstructure.EmbeddedLayerBuilder import EmbeddedLayerBuilder


class SiameseModelBuilder:

    def __init__(self):
        self.__embedded_layer_builder = EmbeddedLayerBuilder()

    def get_siamese_model(self, input_shape):

        embedded_layer = self.__embedded_layer_builder.build_layer(input_shape)

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
