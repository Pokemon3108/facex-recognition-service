from keras import Input, Model, Sequential, layers
from keras.applications.xception import Xception
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import tensorflow as tf


class EmbeddingLayerBuilder:

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
