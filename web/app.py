import io

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug=True)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/upload', methods=['POST'])
def upload():
    pic = request.files['pic']
    image_bytes = Image.open(io.BytesIO(pic.read()))

    opencvImage = cv2.cvtColor(np.array(image_bytes), cv2.COLOR_RGB2BGR)

    if not pic:
        return 'No pic uploaded!', 400

    filename = secure_filename(pic.filename)
    mimetype = pic.mimetype

    if not filename or not mimetype:
        return 'Bad upload!', 400

    return 'Img Uploaded!', 200
