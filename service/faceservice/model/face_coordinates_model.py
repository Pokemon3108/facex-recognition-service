class FaceCoordinatesModel:
    def __init__(self, x, y, weight, height):
        self.x = x.item()
        self.y = y.item()
        self.weight = weight.item()
        self.height = height.item()