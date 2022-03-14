import io
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request

from service.faceservice.facedetector import FaceDetector
from service.modelconverter import buildModelsFaceCoordinates

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug=True)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api/v1/detection', methods=['POST'])
def upload():
    pic = request.files['pic']
    image_bytes = Image.open(io.BytesIO(pic.read()))

    opencvImage = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

    faceDetector = FaceDetector()
    coordinates = faceDetector.buildFaceCoordinatesFromImageBytes(opencvImage)
    faceModels = buildModelsFaceCoordinates(coordinates)

    return json.dumps([fm.__dict__ for fm in faceModels]), 200
