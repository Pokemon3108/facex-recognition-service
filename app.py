import io
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request

from service.databaseservice.face_data_dao import get_all_faces
from service.faceservice.detection.face_detector import FaceDetector
from service.faceservice.model_converter import build_models_face_coordinates, file_storage_to_opencv_image, \
    extract_faces_arr_from_face_dict
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
    face_models = build_models_face_coordinates(coordinates)

    return json.dumps([fm.__dict__ for fm in face_models]), 200


@app.route('/api/v1/recognition', methods=['POST'])
def recognize():
    pic = request.files['pic']
    opencv_image = file_storage_to_opencv_image(pic)

    face_recognizer = FaceRecognizer()
    known_faces = get_all_faces()
    known_faces_arr = extract_faces_arr_from_face_dict(known_faces)
    face_recognizer.recognize(opencv_image, known_faces_arr)

    return "Recognition is successful", 200
