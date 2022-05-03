import tensorflow as tf
from keras import Sequential, layers
from keras.applications.xception import Xception

from service.faceservice.recognition.ShapeModel import ShapeModel


class EmbeddedLayerBuilder:

    def build_layer(self, input_shape):
        pretrained_model = Xception(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False,
            pooling='avg',
        )

        for i in range(len(pretrained_model.layers) - 27):
            pretrained_model.layers[i].trainable = False

        encode_model = Sequential([
            pretrained_model,
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ], name="Encode_Model")
        return encode_model


    def copy_weight_to_embedded_layer(self, model):
        layer_encoder = self.build_layer((ShapeModel.get_weight(), ShapeModel.get_height(), ShapeModel.get_channels_amount()))
        i = 0
        for e_layer in model.layers[0].layers[3].layers:
            layer_weight = e_layer.get_weights()
            layer_encoder.layers[i].set_weights(layer_weight)
            i += 1
        return layer_encoder
