import numpy as np
import tensorflow as tf
from keras import metrics

from service.faceservice.recognition.teacher import BatchGenerator
from service.faceservice.recognition.teacher.LearningTester import LearningTester


class Trainer:

    def __init__(self, siamese_network, file_service) -> None:
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")
        self.binary_cross_loss = tf.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.__margin = 0.4
        self.__batch_generator = BatchGenerator(file_service)
        self.__file_service = file_service
        self.__learning_tester = LearningTester(self.__batch_generator, siamese_network)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            # image = data[0]
            # label = data[1]
            # y_pred = self.siamese_network(image)
            #
            # loss = self.binary_cross_loss(label, y_pred)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def train(self, train_list,test_list, root_folder, EPOCHS):

        train_triplet = self.__file_service.create_triplets(root_folder, train_list)
        test_triplet = self.__file_service.create_triplets(root_folder, test_list)

        train_loss = []
        test_metrics = []
        max_acc = 0

        for epoch in range(1, EPOCHS + 1):

            # Training the model on train data
            epoch_loss = []
            for data in self.__batch_generator.get_batch(train_triplet):
                print(111)
                loss = self.train_step(data)
                # loss = self.siamese_network.train_on_batch(data)
                epoch_loss.append(loss)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            train_loss.append(epoch_loss)

            print(f"\nEPOCH: {epoch} \t ")
            print(f"Loss on train    = {epoch_loss:.5f}")

            # Testing the model on test data
            metric = self.__learning_tester.test_on_triplets(test_triplet)
            test_metrics.append(metric)
            accuracy = metric[0]

            # Saving the model weights
            if accuracy >= max_acc:
                self.siamese_network.save_weights("siamese_model")
                max_acc = accuracy



    def test_on_triplets(self, test_triplet, batch_size=256):
        pos_scores, neg_scores = [], []

        for data in self.__batch_generator.get_batch(test_triplet, batch_size=batch_size):
            prediction = self.siamese_network.predict(data)
            pos_scores += list(prediction[0])
            neg_scores += list(prediction[1])

        accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
        ap_mean = np.mean(pos_scores)
        an_mean = np.mean(neg_scores)
        ap_stds = np.std(pos_scores)
        an_stds = np.std(neg_scores)

        print(f"Accuracy on test = {accuracy:.5f}")
        return (accuracy, ap_mean, an_mean, ap_stds, an_stds)


    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.__margin, 0.0)
        return loss


