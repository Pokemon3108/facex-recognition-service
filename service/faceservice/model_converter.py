import pickle

import pymongo
from bson import Binary

from service.databaseservice.face_bytes_model import FaceBytesModel
from service.faceservice.model.face_coordinates_model import FaceCoordinatesModel


def build_models_face_coordinates(coordinates):
    face_models = []
    for (x, y, w, h) in coordinates:
        face_model = FaceCoordinatesModel(x, y, w, h)
        face_models.append(face_model)
    return face_models


def dictionary_faces_to_face_bytes_model(dict):
    face_bytes_model_arr = []
    for key, value in dict.items():
        face_bytes_model = FaceBytesModel(key, Binary(pickle.dumps(value, protocol=2)))
        face_bytes_model_arr.append(face_bytes_model)
    return face_bytes_model_arr
