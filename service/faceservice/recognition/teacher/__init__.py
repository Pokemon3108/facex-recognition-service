from injectable import load_injection_container

from service.faceservice.recognition.networkstructure.SiameseModelBuilder import SiameseModelBuilder
from service.faceservice.recognition.teacher.BatchGenerator import BatchGenerator
from service.faceservice.recognition.teacher.FileService import FileService, ROOT
from service.faceservice.recognition.teacher.Trainer import Trainer

if __name__ == '__main__':
    load_injection_container()
    file_service = FileService()
    train_list, test_list = file_service.split_dataset()

    siamese_model_builder = SiameseModelBuilder()

    siamese_model = siamese_model_builder.get_siamese_model()
    siamese_model.compile()

    trainer = Trainer(siamese_model)

    EPOCHS = 50
    trainer.train(train_list, test_list, ROOT, EPOCHS)
