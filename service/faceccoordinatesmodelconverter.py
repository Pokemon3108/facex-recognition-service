from service.faceservice.model.facemodel import FaceModel


def build_models_face_coordinates(coordinates):
    face_models = []
    for (x, y, w, h) in coordinates:
        face_model = FaceModel(x, y, w, h)
        face_models.append(face_model)
    return face_models
