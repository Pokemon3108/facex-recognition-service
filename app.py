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
from service.databaseservice.FaceDbService import FaceDbService
from service.faceservice.FaceBytesService import FaceBytesService
from service.faceservice.FaceValidator import FaceValidator
from service.faceservice.ModelConverter import ModelConverter
from service.faceservice.detection.FaceDetector import FaceDetector
from service.faceservice.recognition.Classifier import encoder_model
from service.faceservice.recognition.DistanceService import DistanceService
from service.faceservice.recognition.FaceRecognizer import FaceRecognizer
from service.faceservice.recognition.ShapeModel import ShapeModel
from service.faceservice.recognition.networkstructure.EmbeddedLayerBuilder import EmbeddedLayerBuilder
from service.faceservice.recognition.networkstructure.SiameseModel import SiameseModel
from service.faceservice.recognition.networkstructure.SiameseModelBuilder import SiameseModelBuilder
from service.imageservice.ImageProcessor import ImageProcessor
from web.response_model.Message import Message
from web.response_model.RecognizedNameModel import RecognizedNameModel

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug=True)

load_injection_container()
face_db_service = FaceDbService()
face_bytes_service = FaceBytesService()
face_recognizer = FaceRecognizer()
distance_service = DistanceService()
model_converter = ModelConverter()
face_validator = FaceValidator()
image_processor = ImageProcessor()
face_detector = FaceDetector()


@app.route('/api/v1/detection', methods=['POST'])
def detect():
    pic = request.files['pic']
    image_bytes = Image.open(io.BytesIO(pic.read()))

    opencv_image = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

    face_models = face_detector.build_face_coordinates_model_from_opencv_image(opencv_image)
    return json.dumps([fm.__dict__ for fm in face_models]), 200


@app.route('/api/v1/recognition/group/<group>', methods=['POST'])
def recognize(group):
    pic = request.files['pic']
    opencv_image = model_converter.file_storage_to_opencv_image(pic)
    validate(opencv_image)

    known_faces = face_bytes_service.read_all_faces_by_group(group)
    known_faces_arr = list(map(lambda model: model_converter.extract_np_faces_bytes_from_model(model), known_faces))

    opencv_processed_image = image_processor.extract_resized_face(opencv_image)

    distance_arr = face_recognizer.recognize(opencv_processed_image, known_faces_arr)

    if distance_service.arr_consists_of_ones(distance_arr):
        return Message("Face has not been recognized.").__dict__, 200

    min_distance_index = distance_service.get_smallest_distance_index(distance_arr)
    recognized_name = known_faces[min_distance_index].name

    recognized_name_model = RecognizedNameModel(recognized_name, True)

    return jsonify(recognized_name_model.__dict__), 200


@app.route('/api/v1/recognition/user/<name>/group/<group>', methods=['POST'])
def check_if_user_is_real(name, group):
    pic = request.files['pic']

    opencv_image = model_converter.file_storage_to_opencv_image(pic)
    validate(opencv_image)
    opencv_processed_image = image_processor.extract_resized_face(opencv_image)

    face_bytes_model_db = face_bytes_service.read_face_by_username_and_group(name, group)
    distance_arr = face_recognizer.recognize_pair_opencv_img(
        opencv_processed_image, model_converter.extract_np_faces_bytes_from_model(face_bytes_model_db))

    face_matches_username = distance_service.check_if_distance_is_small(distance_arr[0])

    recognized_name_model = RecognizedNameModel(name, face_matches_username)

    return recognized_name_model.__dict__, 200


@app.route('/api/v1/recognition/pair', methods=['POST'])
def compare_two_images():
    pic1 = request.files['pic1']
    pic2 = request.files['pic2']

    opencv_image1 = model_converter.file_storage_to_opencv_image(pic1)
    validate(opencv_image1)
    opencv_image2 = model_converter.file_storage_to_opencv_image(pic2)
    validate(opencv_image2)

    opencv_processed_image1 = image_processor.extract_resized_face(opencv_image1)
    opencv_processed_image2 = image_processor.extract_resized_face(opencv_image2)
    distance_arr = face_recognizer.recognize_pair_opencv_img(opencv_processed_image1, opencv_processed_image2)

    is_one_person = distance_service.check_if_distance_is_small(distance_arr[0])

    return {'isOnePerson': is_one_person}, 200


@app.route('/api/v1/user/<name>/group/<group>', methods=['POST'])
def upload_user_face(name, group):
    pic = request.files['pic']
    opencv_image = model_converter.file_storage_to_opencv_image(pic)
    validate(opencv_image)

    processed_image = image_processor.extract_resized_face(opencv_image)

    model = FaceBytesModel(name, model_converter.opencv_image_to_bytes(processed_image), group)

    face_bytes_service.save_face(model)
    return Message("Face was successfully loaded.").__dict__, 201


@app.route('/api/v1/user/<name>/group/<group>', methods=['PATCH'])
def update_user_face(name, group):
    pic = request.files['pic']
    opencv_image = model_converter.file_storage_to_opencv_image(pic)
    validate(opencv_image)

    opencv_processed_image = image_processor.extract_resized_face(opencv_image)

    model = FaceBytesModel(name, model_converter.opencv_image_to_bytes(opencv_processed_image), group)

    face_bytes_service.update_face(model)
    return Message("Face was successfully updated.").__dict__, 200


def validate(opencv_image):
    if not face_validator.is_one_face_on_image(opencv_image):
        raise ManyFacesException("There are no faces on image or more than 1.")


import web.error_processor