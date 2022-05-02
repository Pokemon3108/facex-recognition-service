from exception.duplicate_username_exception import DuplicateUsernameException
from service.databaseservice.face_bytes_model import FaceBytesModel
from service.databaseservice.face_db_service import FaceDbService


class FaceBytesService:
    face_db_service = FaceDbService()

    def read_face_by_username(self, username):
        return self.face_db_service.get_face_by_username(username)

    def read_all_faces(self):
        return self.face_db_service.get_all_faces()

    def save_face(self, face_model):
        if self.read_face_by_username(face_model.name) is None:
            return self.face_db_service.save_known_face(face_model.__dict__)
        else:
            raise DuplicateUsernameException("Person with this name already exists.")

    def update_face(self, face_model: FaceBytesModel):
        return self.face_db_service.update_face_bytes(face_model)
