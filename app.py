import io
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request

from exception.not_found_exception import NotFoundException
from service.databaseservice.face_db_service import FaceDbService
from service.faceservice.detection.face_detector import FaceDetector
from service.faceservice.model_converter import ModelConverter
from service.faceservice.recognition.classifier import Classifier
from service.faceservice.recognition.distance_service import DistanceService
from service.faceservice.recognition.face_recognizer import FaceRecognizer
from service.imageservice.image_processor import ImageProcessor

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug=True)

face_data_folder_path = "/archive/User Faces"


@app.route('/')
def ping():
    return 'Hello, World!'


@app.route('/api/v1/detection', methods=['POST'])
def detect():
    pic = request.files['pic']
    image_bytes = Image.open(io.BytesIO(pic.read()))

    opencv_image = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

    face_detector = FaceDetector()
    coordinates = face_detector.build_face_coordinates_from_image_bytes(opencv_image)

    model_converter = ModelConverter()
    face_models = model_converter.build_models_face_coordinates(coordinates)

    return json.dumps([fm.__dict__ for fm in face_models]), 200


@app.route('/api/v1/recognition', methods=['POST'])
def recognize():
    model_converter = ModelConverter()

    pic = request.files['pic']
    opencv_image = process_image_from_request(pic)

    face_db_service = FaceDbService()
    known_faces = face_db_service.get_all_faces()
    known_faces_arr = list(map(lambda model: model_converter.extract_faces_bytes_from_model(model), known_faces))

    face_recognizer = FaceRecognizer()
    distance_arr = face_recognizer.recognize(opencv_image, known_faces_arr)

    distance_service = DistanceService()
    min_distance_index = distance_service.get_smallest_distance_index(distance_arr)
    recognized_name = known_faces[min_distance_index].name

    return recognized_name, 200


@app.route('/api/v1/recognition/user/<name>', methods=['POST'])
def check_if_user_is_real(name):
    pic = request.files['pic']
    opencv_image = process_image_from_request(pic)

    model_converter = ModelConverter()
    face_db_service = FaceDbService()
    face_bytes_model = face_db_service.get_face_by_username(name)
    if face_bytes_model is None:
        raise NotFoundException("No face was found.")
    face_bytes_np = [model_converter.extract_faces_bytes_from_model(face_bytes_model)]

    face_recognizer = FaceRecognizer()
    distance_arr = face_recognizer.recognize(opencv_image, face_bytes_np)

    distance_service = DistanceService()
    face_matches_username = distance_service.check_if_distance_is_small(distance_arr[0])

    return face_matches_username.__str__(), 200


def process_image_from_request(pic):
    model_converter = ModelConverter()
    opencv_image = model_converter.file_storage_to_opencv_image(pic)

    classifier = Classifier()
    image_processor = ImageProcessor()
    return image_processor.resize(opencv_image, classifier.get_weight(), classifier.get_height())


import web.error_processor
