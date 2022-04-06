from service.databaseservice.connector import MongoConnector
from service.databaseservice.face_bytes_model import FaceBytesModel


class FaceDbService:
    connector = MongoConnector(db_name='facex',
                               collection_name='faceData',
                               hostname="localhost",
                               port=27017)

    def save_known_face(self, obj_to_save):
        self.connector.save(obj_to_save)

    def get_face_by_username(self, username) -> FaceBytesModel | None:
        search_criteria = {'name': username}
        doc = self.connector.search(search_criteria)
        if doc is None:
            return None
        return FaceBytesModel(doc['name'], doc['bytes'])

    def get_all_faces(self) -> list[FaceBytesModel]:
        cursor = self.connector.read_all()
        faces_model_arr = []
        for doc in cursor:
            face_name = doc['name']
            faces_bytes_str = doc['bytes']

            # np_faces_bytes = np.load(BytesIO(faces_bytes_str), allow_pickle=True)
            faces_model_arr.append(FaceBytesModel(face_name, faces_bytes_str))
        return faces_model_arr
