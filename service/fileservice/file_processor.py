import os

import cv2
from pathlib import Path


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_known_faces(local_path):
    folders, path_to_set = list_folder_by_path(local_path)
    user_name_to_face_dict = {}
    for folder in folders:
        full_path_to_person_face = path_to_set + "\\" + folder
        person_face_list = os.listdir(full_path_to_person_face)
        full_paths = list(map(lambda name: os.path.join(full_path_to_person_face, name), person_face_list))
        if len(full_paths) > 0:
            image = read_image(full_paths[0])
            user_name_to_face_dict[folder] = image

    return user_name_to_face_dict


def list_folder_by_path(local_path):
    path_to_root = Path(__file__).parent.parent.parent
    path_to_set = str(path_to_root) + local_path
    return os.listdir(path_to_set), path_to_set
