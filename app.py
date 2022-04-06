import io
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request

from exception.many_faces_exception import ManyFacesException
from exception.not_found_exception import NotFoundException
from service.databaseservice.face_bytes_model import FaceBytesModel
from service.databaseservice.face_db_service import FaceDbService
from service.faceservice.detection.face_detector import FaceDetector
from service.faceservice.model_converter import ModelConverter
from service.faceservice.recognition.classifier import Classifier
from service.faceservice.recognition.distance_service import DistanceService
from service.faceservice.recognition.face_recognizer import FaceRecognizer
from service.faceservice.validator import FaceValidator
from service.imageservice.image_processor import ImageProcessor
from web.response_model.message_model import Message
from web.response_model.recognized_name_model import RecognizedNameModel

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug=True)

face_data_folder_path = "/archive/User Faces"


@app.route('/')
def ping():
    return 'Hello, World!'


@app.route('/api/v1/detection', methods=['GET'])
def detect():
    pic = request.files['pic']
    image_bytes = Image.open(io.BytesIO(pic.read()))

    opencv_image = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

    face_detector = FaceDetector()
    coordinates = face_detector.build_face_coordinates_from_opencv_image(opencv_image)

    model_converter = ModelConverter()
    face_models = model_converter.build_models_face_coordinates(coordinates)

    return json.dumps([fm.__dict__ for fm in face_models]), 200


@app.route('/api/v1/recognition', methods=['GET'])
def recognize():
    model_converter = ModelConverter()
    face_validator = FaceValidator()

    pic = request.files['pic']
    opencv_image = resize(pic)

    if face_validator.is_one_face_on_image(opencv_image):
        raise ManyFacesException("There are no faces on image or more than 1.")

    face_db_service = FaceDbService()
    known_faces = face_db_service.get_all_faces()
    known_faces_arr = list(map(lambda model: model_converter.extract_np_faces_bytes_from_model(model), known_faces))

    face_recognizer = FaceRecognizer()
    distance_arr = face_recognizer.recognize(opencv_image, known_faces_arr)

    distance_service = DistanceService()
    min_distance_index = distance_service.get_smallest_distance_index(distance_arr)
    recognized_name = known_faces[min_distance_index].name

    recognized_name_model = RecognizedNameModel(recognized_name, True)

    return json.dumps(recognized_name_model.__dict__), 200


@app.route('/api/v1/recognition/user/<name>', methods=['GET'])
def check_if_user_is_real(name):
    pic = request.files['pic']
    opencv_image = resize(pic)

    face_validator = FaceValidator()
    if not face_validator.is_one_face_on_image(opencv_image):
        raise ManyFacesException("There are no faces on image or more than 1.")

    model_converter = ModelConverter()
    face_db_service = FaceDbService()
    face_bytes_model = face_db_service.get_face_by_username(name)
    if face_bytes_model is None:
        raise NotFoundException("No face was found.")
    face_bytes_np = [model_converter.extract_np_faces_bytes_from_model(face_bytes_model)]

    face_recognizer = FaceRecognizer()
    distance_arr = face_recognizer.recognize(opencv_image, face_bytes_np)

    distance_service = DistanceService()
    face_matches_username = distance_service.check_if_distance_is_small(distance_arr[0])

    recognized_name_model = RecognizedNameModel(name, face_matches_username)

    return recognized_name_model.__dict__, 200


@app.route('/api/v1/user/<name>', methods=['POST'])
def upload_user_face(name):
    pic = request.files['pic']
    opencv_resized_image = resize(pic)

    model_converter = ModelConverter()
    model = FaceBytesModel(name, model_converter.opencv_image_to_bytes(opencv_resized_image))

    face_db_service = FaceDbService()
    face_db_service.save_known_face(model.__dict__)
    return Message("Face was successfully loaded").__dict__, 200


def resize(pic):
    model_converter = ModelConverter()
    opencv_image = model_converter.file_storage_to_opencv_image(pic)

    classifier = Classifier()
    image_processor = ImageProcessor()
    return image_processor.resize(opencv_image, classifier.get_weight(), classifier.get_height())


import web.error_processor
