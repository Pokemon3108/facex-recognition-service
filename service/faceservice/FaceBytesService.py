from exception.DuplicateUsernameException import DuplicateUsernameException
from exception.NotFoundException import NotFoundException
from service.databaseservice.FaceBytesModel import FaceBytesModel
from service.databaseservice.FaceDbService import FaceDbService


class FaceBytesService:
    face_db_service = FaceDbService()

    def read_face_by_username(self, username):
        return self.face_db_service.get_face_by_username(username)

    def read_all_faces_by_group(self, group):
        faces = self.face_db_service.get_faces_by_group(group)
        if faces is None:
            raise NotFoundException(f'The group {group} not exists.')

    def save_face(self, face_model):
        if self.read_face_by_username(face_model.name) is None:
            return self.face_db_service.save_known_face(face_model.__dict__)
        else:
            raise DuplicateUsernameException("Person with this name already exists.")

    def update_face(self, face_model: FaceBytesModel):
        return self.face_db_service.update_face_bytes(face_model)
