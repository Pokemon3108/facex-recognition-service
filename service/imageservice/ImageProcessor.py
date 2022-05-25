import cv2
from injectable import Autowired, autowired

from service.faceservice.recognition.ShapeModel import ShapeModel
from service.faceservice.detection.FaceDetector import FaceDetector


class ImageProcessor:

    @autowired
    def __init__(self, face_detector : Autowired(FaceDetector)):
        self.__face_detector = face_detector

    def extract_resized_face(self, opencv_image):
        faces = self.extract_face(opencv_image)
        return self.resize(faces[0], ShapeModel.get_width(), ShapeModel.get_height())

    def resize(self, cv_image, width, height):
        return cv2.resize(cv_image, (width, height))

    def extract_face(self, cv_image):
        coordinates_models = self.__face_detector.build_face_coordinates_model_from_opencv_image(cv_image)
        faces_list = []
        for model in coordinates_models:
            face = cv_image[model.y: model.y + model.height, model.x: model.x + model.width]
            faces_list.append(face)
        return faces_list

