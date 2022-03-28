import io
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request

from service.databaseservice.face_db_service import FaceDbService
from service.faceservice.detection.face_detector import FaceDetector
from service.faceservice.model_converter import ModelConverter
from service.faceservice.recognition.face_recognizer import FaceRecognizer

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug=True)

face_data_folder_path = "/archive/User Faces"


# @app.before_first_request
# def run_on_start():
#     face_data_folder_path = "/archive/User Faces"
#     known_faces = get_known_faces(face_data_folder_path)
#     known_faces_model = dictionary_faces_to_face_bytes_model(known_faces)
#     for f in known_faces_model:
#         save_known_face(f.__dict__)


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
    opencv_image = model_converter.file_storage_to_opencv_image(pic)

    face_db_service = FaceDbService()
    known_faces = face_db_service.get_all_faces()
    known_faces_arr = list(map(lambda model: model_converter.extract_faces_bytes_from_model(model), known_faces))

    face_recognizer = FaceRecognizer()
    face_recognizer.recognize(opencv_image, known_faces_arr)

    return "Recognition is successful", 200


@app.route('/api/v1/recognition/user/<name>', methods=['POST'])
def check_if_user_is_real(name):
    model_converter = ModelConverter()
    pic = request.files['pic']
    opencv_image = model_converter.file_storage_to_opencv_image(pic)

    face_db_service = FaceDbService()
    face_bytes_model = face_db_service.get_face_by_username(name)
    face_bytes_np = [model_converter.extract_faces_bytes_from_model(face_bytes_model)]

    face_recognizer = FaceRecognizer()
    face_recognizer.recognize(opencv_image, face_bytes_np)

    return "Recognition is successful", 200
