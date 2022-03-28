import numpy as np
from keras.applications.inception_v3 import preprocess_input

from service.faceservice.recognition.classifier import Classifier


class FaceRecognizer:

    classifier = Classifier()

    def recognize(self, opencv_image, known_faces_arr):
        image_for_recognize_np = np.array([opencv_image])
        pre_process_input_for_recognize = preprocess_input(image_for_recognize_np)
        known_faces_np = np.array(known_faces_arr)
        pre_process_input_known_faces = preprocess_input(known_faces_np)
        return self.classifier.classify_images(pre_process_input_for_recognize, pre_process_input_known_faces)
