class FaceCoordinatesModel:
    def __init__(self, x, y, width, height):
        self.x = x.item()
        self.y = y.item()
        self.width = width.item()
        self.height = height.item()