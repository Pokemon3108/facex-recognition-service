import numpy as np
from injectable import injectable
from keras.applications.inception_v3 import preprocess_input

from service.faceservice.recognition.Classifier import Classifier


@injectable
class FaceRecognizer:

    def __init__(self):
        self.__classifier = Classifier()

    def recognize(self, opencv_image, known_faces_arr):
        image_for_recognize_np = np.array([opencv_image])
        pre_process_input_for_recognize = preprocess_input(image_for_recognize_np)
        known_faces_np = np.array(known_faces_arr)
        pre_process_input_known_faces = preprocess_input(known_faces_np)
        return self.__classifier.classify_images(pre_process_input_for_recognize, pre_process_input_known_faces)

    def recognize_pair_opencv_img(self, opencv_image1, opencv_image2):
        # cv2.imshow('image1', opencv_image1)
        # cv2.waitKey(0)
        # cv2.imshow('image2', opencv_image2)
        # cv2.waitKey(0)
        image_for_recognize_np1 = np.array([opencv_image1])
        pre_process_input_for_recognize1 = preprocess_input(image_for_recognize_np1)
        image_for_recognize_np2 = np.array([opencv_image2])
        pre_process_input_for_recognize2 = preprocess_input(image_for_recognize_np2)
        return self.__classifier.classify_images(pre_process_input_for_recognize1, pre_process_input_for_recognize2)
