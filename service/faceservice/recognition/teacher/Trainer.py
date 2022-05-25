import numpy as np
import tensorflow as tf
from injectable import Autowired, autowired
from keras import metrics

from service.faceservice.recognition.teacher import BatchGenerator, FileService
from service.faceservice.recognition.teacher.LearningTester import LearningTester


class Trainer:

    @autowired
    def __init__(self, siamese_network,
                 file_service: Autowired(FileService)) -> None:
        self.siamese_network = siamese_network
        # self.loss_tracker = metrics.Mean(name="loss")
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.__margin = 0.4
        self.__batch_generator = BatchGenerator(file_service)
        self.__file_service = file_service
        self.__learning_tester = LearningTester(self.__batch_generator, siamese_network)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        # self.loss_tracker.update_state(loss)
        return sum(loss) / len(loss)

    def train(self, train_list, test_list, root_folder, EPOCHS):

        train_triplet = self.__file_service.create_triplets(root_folder, train_list)
        test_triplet = self.__file_service.create_triplets(root_folder, test_list)

        train_loss = []
        test_acc = []
        max_acc = 0

        for epoch in range(1, EPOCHS + 1):

            # Training the model on train data
            epoch_loss = []
            for data in self.__batch_generator.get_batch(train_triplet):
                loss = self.train_step(data)
                # loss = self.siamese_network.train_on_batch(data)
                epoch_loss.append(loss)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            train_loss.append(epoch_loss)

            print(f"\nEPOCH: {epoch} \t ")
            print(f"Loss on train    = {epoch_loss:.5f}")

            # Testing the model on test data
            accuracy = self.__learning_tester.test_on_triplets(test_triplet)
            test_acc.append(accuracy)

            # Saving the model weights
            if accuracy >= max_acc:
                self.siamese_network.save_weights("siamese_model")
                max_acc = accuracy
            else:
                self.siamese_network.load_weights("siamese_model")

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.__margin, 0.0)
        return loss
