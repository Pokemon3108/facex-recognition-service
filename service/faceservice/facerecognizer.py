import os
from pathlib import Path

import numpy as np
from keras.applications.inception_v3 import preprocess_input

from service.faceservice.neuralnetworkmodel.classifier import classify_images
from service.fileservice.fileprocessor import read_image


class FaceRecognizer:

    def get_known_faces(self, local_path):
        path_to_root = Path(__file__).parent.parent.parent
        path_to_set = str(path_to_root) + local_path
        folders = os.listdir(path_to_set)
        faces = []
        for folder in folders:
            full_path_to_person_face = path_to_set + "\\" + folder
            person_face_list = os.listdir(full_path_to_person_face)
            full_paths = list(map(lambda name: os.path.join(full_path_to_person_face, name), person_face_list))
            if len(full_paths) > 0:
                image = read_image(full_paths[0])
                faces.append(image)

        return faces

    def recognize(self, opencv_image, known_faces):
        image_for_recognize_np = np.array([opencv_image])
        pre_process_input_for_recognize = preprocess_input(image_for_recognize_np)
        known_faces_np = np.array(known_faces)
        pre_process_input_known_faces = preprocess_input(known_faces_np)
        vect = classify_images(pre_process_input_for_recognize, pre_process_input_known_faces)
        print(vect)
