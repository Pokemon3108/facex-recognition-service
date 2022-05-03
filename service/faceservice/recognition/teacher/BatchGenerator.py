import os
import random

import cv2
import numpy as np
import tensorflow as tf
from injectable import Autowired, injectable
from keras.applications.inception_v3 import preprocess_input

from service.faceservice.recognition.teacher import FileService


@injectable
class BatchGenerator:

    def __init__(self, file_service : Autowired(FileService)) -> None:
        self.__file_service = file_service

    def get_batch(self, triplet_list, batch_size=256, preprocess=True):
        batch_steps = len(triplet_list)

        for i in range(batch_steps + 1):
            anchor = []
            positive = []
            negative = []

            j = i * batch_size
            while j < (i + 1) * batch_size and j < len(triplet_list):
                a, p, n = triplet_list[j]
                anchor.append(self.__file_service.read_image(a))
                positive.append(self.__file_service.read_image(p))
                negative.append(self.__file_service.read_image(n))
                j += 1

            anchor = np.array(anchor)
            positive = np.array(positive)
            negative = np.array(negative)

            if preprocess:
                anchor = preprocess_input(anchor)
                positive = preprocess_input(positive)
                negative = preprocess_input(negative)

            yield ([anchor, positive, negative])

    def create_triplets(self, directory, folder_list, max_files=10):
        triplets = []
        folders = list(folder_list.keys())

        for folder in folders:
            path = os.path.join(directory, folder)
            files = list(os.listdir(path))[:max_files]
            num_files = len(files)

            for i in range(num_files - 1):
                for j in range(i + 1, num_files):
                    anchor = (folder, f"{i}.jpg")
                    positive = (folder, f"{j}.jpg")

                    neg_folder = folder
                    while neg_folder == folder:
                        neg_folder = random.choice(folders)
                    neg_file = random.randint(0, folder_list[neg_folder] - 1)
                    negative = (neg_folder, f"{neg_file}.jpg")

                    triplets.append((anchor, positive, negative))

        random.shuffle(triplets)
        return triplets

    def preprocess(self, byte_img):

        # Load in the image
        processed_img = cv2.resize(byte_img, (100, 100))
        # Scale image to be between 0 and 1
        processed_img = processed_img / 255.0

        # Return image
        return processed_img
