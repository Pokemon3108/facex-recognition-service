import tensorflow as tf
from keras import layers, metrics
from keras.models import Model

from service.faceservice.recognition.distance_layer import DistanceLayer
from service.faceservice.recognition.encoder import  EncoderBuilder


class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()

        self.__margin = margin
        self.siamese_network = siamese_network
        self.__loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.__loss_tracker.update_state(loss)
        return {"loss": self.__loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.__loss_tracker.update_state(loss)
        return {"loss": self.__loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.__loss_tracker]


def get_siamese_network(input_shape=(128, 128, 3)):
    encoder_builder = EncoderBuilder()
    encoder = encoder_builder.get_encoder(input_shape)

    # Input Layers for the images
    anchor_input = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")

    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )

    # Creating the Model
    siamese_network_model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="Siamese_Network"
    )
    return siamese_network_model
