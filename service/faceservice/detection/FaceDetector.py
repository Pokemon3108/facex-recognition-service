import cv2
from injectable import injectable


@injectable
class FaceDetector:
    __cascade_path = "./cascade/haarcascade_frontalface_default.xml"

    def __init__(self) -> None:
        self.__face_cascade = cv2.CascadeClassifier(self.__cascade_path)

    def build_face_coordinates(self, image_path):
        image = cv2.imread(image_path)
        return self.build_face_coordinates_from_opencv_image(image)

    def build_face_coordinates_from_opencv_image(self, opencv_image):
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        return self.__face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
