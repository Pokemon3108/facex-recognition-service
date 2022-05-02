from service.faceservice.recognition.teacher.FileService import FileService

ROOT = "../archive/Extracted Faces/Extracted Faces"

if __name__ == '__main__':
    file_service = FileService(ROOT)
    train_list, test_list = file_service.split_dataset(ROOT)

    shape=(128,128,3)

    siamese_model = SiameseModel().get_siamese_model(shape)
    siamese_model.compile()

    batch_generator = BatchGenerator(file_service)
    trainer = Trainer(siamese_model, file_service)

    EPOCHS = 50
    trainer.train(train_list, test_list, ROOT, EPOCHS)
