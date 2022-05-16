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


