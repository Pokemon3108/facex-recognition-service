from service.faceservice.detection.FaceDetector import FaceDetector


class FaceValidator:

    def __init__(self):
        self.__detector = FaceDetector()

    def is_one_face_on_image(self, opencv_image):
        coordinates = self.__detector.build_face_coordinates_from_opencv_image(opencv_image)
        return len(coordinates) == 1
