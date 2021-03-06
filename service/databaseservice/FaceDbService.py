from injectable import injectable

from service.databaseservice.MongoConnector import MongoConnector
from service.databaseservice.FaceBytesModel import FaceBytesModel

@injectable
class FaceDbService:

    def __init__(self):
        self.__connector = MongoConnector(db_name='facex',
                                     collection_name='faceData',
                                     hostname="localhost",
                                     port=27017)

    def save_known_face(self, obj_to_save):
        self.__connector.save(obj_to_save)

    def get_face_by_username_and_group(self, username, group) -> FaceBytesModel | None:
        search_criteria = {'name': username, 'group': group}
        doc = self.__connector.search(search_criteria)
        if doc is None:
            return None
        return FaceBytesModel(doc['name'], doc['bytes'], doc['group'])

    def get_faces_by_group(self, group) -> list[FaceBytesModel] | None:
        search_criteria = {'group': str(group)}
        cursor = self.__connector.search_all(search_criteria)
        if cursor is None:
            return None
        faces_model_arr = []
        for doc in cursor:
            faces_model_arr.append(FaceBytesModel(doc['name'], doc['bytes'], doc['group']))
        return faces_model_arr

    def update_face_bytes(self, face_model: FaceBytesModel):
        update_criteria = {'name': face_model.name}, {'$set': {"bytes": face_model.bytes}}
        doc = self.__connector.update(update_criteria[0], update_criteria[1])
        if doc is None:
            return None
        return FaceBytesModel(doc['name'], doc['bytes'], doc['group'])
