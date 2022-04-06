from service.faceservice.detection.face_detector import FaceDetector


class FaceValidator:
    detector = FaceDetector()

    def is_one_face_on_image(self, opencv_image):
        coordinates = self.detector.build_face_coordinates_from_opencv_image(opencv_image)
        return len(coordinates) == 1
