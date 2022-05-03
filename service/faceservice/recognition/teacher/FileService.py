import os
from pathlib import Path

import cv2
import random

from injectable import injectable


@injectable
class FileService:

    def __init__(self, root_folder) -> None:
        super().__init__()
        self.__root = root_folder

    def read_image(self, index):
        path = os.path.join(self.__root, index[0], index[1])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def split_dataset(self, directory):
        folders = os.listdir(directory)
        num_train = len(folders)

        random.shuffle(folders)

        train_list, test_list = {}, {}

        # Creating Train-list
        for folder in folders[:num_train]:
            num_files = len(os.listdir(os.path.join(directory, folder)))
            train_list[folder] = num_files

        # Creating Test-list
        for folder in folders[num_train:]:
            num_files = len(os.listdir(os.path.join(directory, folder)))
            test_list[folder] = num_files

        return train_list, test_list

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
