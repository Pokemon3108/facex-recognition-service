import os
from pathlib import Path

import numpy as np
from keras.applications.inception_v3 import preprocess_input

from service.faceservice.neuralnetworkmodel.classifier import classify_images
from service.fileservice.file_processor import read_image


class FaceRecognizer:

    def recognize(self, opencv_image, known_faces):
        image_for_recognize_np = np.array([opencv_image])
        pre_process_input_for_recognize = preprocess_input(image_for_recognize_np)
        known_faces_np = np.array(known_faces)
        pre_process_input_known_faces = preprocess_input(known_faces_np)
        vect = classify_images(pre_process_input_for_recognize, pre_process_input_known_faces)
        print(vect)
