from keras import Model


class SiameseModel(Model):

    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network

    def call(self, inputs):
        return self.siamese_network(inputs)
