import numpy as np


class LearningTester:

    def __init__(self, batch_generator, siamese_network):
        self.__batchGenerator = batch_generator
        self.__siamese_network = siamese_network

    def test_on_triplets(self, test_triplet, batch_size=256):
        pos_scores, neg_scores = [], []

        for data in self.__batchGenerator.get_batch(test_triplet, batch_size=batch_size):
            prediction = self.__siamese_network.predict(data)
            pos_scores += list(prediction[0])
            neg_scores += list(prediction[1])

        accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
        ap_mean = np.mean(pos_scores)
        an_mean = np.mean(neg_scores)
        ap_stds = np.std(pos_scores)
        an_stds = np.std(neg_scores)

        print(f"Accuracy on test = {accuracy:.5f}")
        return (accuracy, ap_mean, an_mean, ap_stds, an_stds)