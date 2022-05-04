import io
import pickle

import cv2
import numpy as np
from PIL import Image
from bson import Binary
from io import BytesIO

from injectable import injectable

from service.databaseservice.FaceBytesModel import FaceBytesModel
from service.faceservice.model.FaceCoordinatesModel import FaceCoordinatesModel

@injectable
class ModelConverter:

    def build_models_face_coordinates(self, coordinates):
        face_models = []
        for x, y, w, h in coordinates:
            face_model = FaceCoordinatesModel(x, y, w, h)
            face_models.append(face_model)
        return face_models

    def dictionary_faces_to_face_bytes_model(self, dict):
        face_bytes_model_arr = []
        for key, value in dict.items():
            face_bytes_model = FaceBytesModel(key, Binary(pickle.dumps(value, protocol=2)))
            face_bytes_model_arr.append(face_bytes_model)
        return face_bytes_model_arr

    def file_storage_to_opencv_image(self, file_storage):
        image_bytes = Image.open(io.BytesIO(file_storage.read()))
        return cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

    def opencv_image_to_bytes(self, opencv_image):
        return Binary(pickle.dumps(opencv_image, protocol=2))

    def extract_np_faces_bytes_from_model(self, model):
        return np.load(BytesIO(model.bytes), allow_pickle=True)
