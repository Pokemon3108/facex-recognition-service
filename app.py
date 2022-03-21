import io
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request

from service.faceservice.facedetector import FaceDetector
from service.faceservice.facerecognizer import FaceRecognizer
from service.modelconverter import build_models_face_coordinates

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug=True)

face_recognizer = FaceRecognizer()
face_data_folder_path = "/archive/Extracted Faces"
known_faces = face_recognizer.get_known_faces(face_data_folder_path)


@app.route('/')
def ping():
    return 'Hello, World!'


@app.route('/api/v1/detection', methods=['POST'])
def detect():
    pic = request.files['pic']
    image_bytes = Image.open(io.BytesIO(pic.read()))

    opencvImage = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

    face_detector = FaceDetector()
    coordinates = face_detector.build_face_coordinates_from_image_bytes(opencvImage)
    face_models = build_models_face_coordinates(coordinates)

    return json.dumps([fm.__dict__ for fm in face_models]), 200


@app.route('/api/v1/recognition', methods=['POST'])
def recognize():
    pic = request.files['pic']
    image_bytes = Image.open(io.BytesIO(pic.read()))
    opencvImage = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)
    print(face_recognizer.recognize(opencvImage, known_faces))
    return "111", 200

# import time
#
# import cv2
# import numpy as np
#
# from service.faceservice.facerecognizer import FaceRecognizer
# from service.faceservice.neuralnetworkmodel.classifier import classify_images
# from keras.applications.inception_v3 import preprocess_input
#
# def read_image(path):
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image
#
#
#
# ROOT1 = "./archive/Extracted Faces/1/0.jpg"
#
# image1content = []
# image1content.append(read_image(ROOT1))
# image1np = np.array(image1content)
# image1 = preprocess_input(image1np)
#
# face_recognizer = FaceRecognizer()
# face_data_folder_path = "/archive/Extracted Faces"
# known_faces = face_recognizer.get_known_faces(face_data_folder_path)
# known_faces_np = np.array(known_faces)
# image2 = preprocess_input(known_faces_np)
#
# start_time = time.time()
# vect = classify_images(image1, image2)
# print(vect)
# print("--- %s seconds ---" % (time.time() - start_time))
