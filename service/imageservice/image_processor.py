import cv2


class ImageProcessor:

    def resize(self, cv_image, width, height):
        return cv2.resize(cv_image, (width, height))
