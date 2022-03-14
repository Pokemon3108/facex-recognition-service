from service.faceservice.facemodel import FaceModel


def buildModelsFaceCoordinates(coordinates):
    faceModels = []
    for (x, y, w, h) in coordinates:
        faceModel = FaceModel(x, y, w, h)
        faceModels.append(faceModel)
    return faceModels
