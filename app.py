import io
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from injectable import load_injection_container

from exception.ManyFacesException import ManyFacesException
from exception.NotFoundException import NotFoundException
from service.databaseservice.FaceBytesModel import FaceBytesModel
from service.faceservice.FaceBytesService import FaceBytesService
from service.faceservice.FaceValidator import FaceValidator
from service.faceservice.ModelConverter import ModelConverter
from service.faceservice.detection.FaceDetector import FaceDetector
from service.faceservice.recognition.DistanceService import DistanceService
from service.faceservice.recognition.FaceRecognizer import FaceRecognizer
from service.faceservice.recognition.ShapeModel import ShapeModel
from service.imageservice.ImageProcessor import ImageProcessor
from web.response_model.Message import Message
from web.response_model.RecognizedNameModel import RecognizedNameModel

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug=True)


@app.before_first_request
def init():
    load_injection_container()


@app.route('/api/v1/detection', methods=['POST'])
def detect():
    pic = request.files['pic']
    image_bytes = Image.open(io.BytesIO(pic.read()))

    opencv_image = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

    face_detector = FaceDetector()
    coordinates = face_detector.build_face_coordinates_from_opencv_image(opencv_image)

    model_converter = ModelConverter()
    face_models = model_converter.build_models_face_coordinates(coordinates)

    return json.dumps([fm.__dict__ for fm in face_models]), 200


@app.route('/api/v1/recognition/<group>', methods=['POST'])
def recognize(group):
    model_converter = ModelConverter()
    face_validator = FaceValidator()

    pic = request.files['pic']
    opencv_image = resize(pic)

    if not face_validator.is_one_face_on_image(opencv_image):
        raise ManyFacesException("There are no faces on image or more than 1.")

    face_bytes_service = FaceBytesService()
    known_faces = face_bytes_service.read_all_faces_by_group(group)
    known_faces_arr = list(map(lambda model: model_converter.extract_np_faces_bytes_from_model(model), known_faces))

    face_recognizer = FaceRecognizer()
    distance_arr = face_recognizer.recognize(opencv_image, known_faces_arr)

    distance_service = DistanceService()
    min_distance_index = distance_service.get_smallest_distance_index(distance_arr)
    recognized_name = known_faces[min_distance_index].name

    recognized_name_model = RecognizedNameModel(recognized_name, True)

    return jsonify(recognized_name_model.__dict__), 200


@app.route('/api/v1/recognition/user/<name>', methods=['POST'])
def check_if_user_is_real(name):
    pic = request.files['pic']
    opencv_image = resize(pic)

    face_validator = FaceValidator()
    if not face_validator.is_one_face_on_image(opencv_image):
        raise ManyFacesException("There are no faces on image or more than 1.")

    model_converter = ModelConverter()
    face_bytes_service = FaceBytesService()
    face_bytes_model = face_bytes_service.read_face_by_username(name)
    if face_bytes_model is None:
        raise NotFoundException("No face of this username was found in database.")
    face_bytes_np = [model_converter.extract_np_faces_bytes_from_model(face_bytes_model)]

    face_recognizer = FaceRecognizer()
    distance_arr = face_recognizer.recognize(opencv_image, face_bytes_np)

    distance_service = DistanceService()
    face_matches_username = distance_service.check_if_distance_is_small(distance_arr[0])

    recognized_name_model = RecognizedNameModel(name, face_matches_username)

    return recognized_name_model.__dict__, 200


@app.route('/api/v1/user/<name>/group/<group>', methods=['POST'])
def upload_user_face(name, group):
    pic = request.files['pic']
    opencv_resized_image = resize(pic)

    model_converter = ModelConverter()
    model = FaceBytesModel(name, model_converter.opencv_image_to_bytes(opencv_resized_image), group)

    face_bytes_service = FaceBytesService()
    face_bytes_service.save_face(model)
    return Message("Face was successfully loaded.").__dict__, 200


@app.route('/api/v1/user/<name>', methods=['PUT'])
def update_user_face(name):
    pic = request.files['pic']
    opencv_resized_image = resize(pic)

    model_converter = ModelConverter()
    model = FaceBytesModel(name, model_converter.opencv_image_to_bytes(opencv_resized_image))

    face_bytes_service = FaceBytesService()
    face_bytes_service.update_face(model)
    return Message("Face was successfully updated.").__dict__, 200


def resize(pic):
    model_converter = ModelConverter()
    opencv_image = model_converter.file_storage_to_opencv_image(pic)

    image_processor = ImageProcessor()
    return image_processor.resize(opencv_image, ShapeModel.get_weight(), ShapeModel.get_height())
