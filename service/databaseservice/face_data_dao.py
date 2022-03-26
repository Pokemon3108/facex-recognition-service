from io import BytesIO

import numpy as np

from service.databaseservice.connector import MongoConnector

connector = MongoConnector(db_name='facex',
                           collection_name='faceData',
                           hostname="localhost",
                           port=27017)


def save_known_face(obj_to_save):
    connector.save(obj_to_save)


def get_all_faces():
    cursor = connector.read_all()
    faces_dict = {}
    for doc in cursor:
        face_name = doc['name']
        faces_bytes_str = doc['bytes']

        np_faces_bytes = np.load(BytesIO(faces_bytes_str), allow_pickle=True)
        faces_dict[face_name] = np_faces_bytes
    return faces_dict
