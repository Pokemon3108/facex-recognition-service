import io
import pickle

import cv2
import numpy as np
from PIL import Image
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


def file_storage_to_opencv_image(file_storage):
    image_bytes = Image.open(io.BytesIO(file_storage.read()))
    return cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)


def extract_faces_arr_from_face_dict(dict):
    face_nd_array = []
    for key, value in dict.items():
        face_nd_array.append(value)
    return face_nd_array
