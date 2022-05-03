from service.faceservice.recognition.networkstructure.SiameseModelBuilder import SiameseModelBuilder
from service.faceservice.recognition.teacher.BatchGenerator import BatchGenerator
from service.faceservice.recognition.teacher.FileService import FileService
from service.faceservice.recognition.teacher.Trainer import Trainer

ROOT = "../archive/Extracted Faces/Extracted Faces"

if __name__ == '__main__':
    file_service = FileService(ROOT)
    train_list, test_list = file_service.split_dataset(ROOT)

    siamese_model_builder = SiameseModelBuilder()

    siamese_model = siamese_model_builder.get_siamese_model()
    siamese_model.compile()

    batch_generator = BatchGenerator(file_service)
    trainer = Trainer(siamese_model, file_service)

    EPOCHS = 50
    trainer.train(train_list, test_list, ROOT, EPOCHS)
