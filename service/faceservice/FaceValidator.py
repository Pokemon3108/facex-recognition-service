from injectable import autowired, Autowired, injectable

from service.faceservice.detection.FaceDetector import FaceDetector

@injectable
class FaceValidator:

    @autowired
    def __init__(self, detector : Autowired(FaceDetector)):
        self.__detector = detector

    def is_one_face_on_image(self, opencv_image):
        coordinates = self.__detector.build_face_coordinates_from_opencv_image(opencv_image)
        return len(coordinates) == 1
