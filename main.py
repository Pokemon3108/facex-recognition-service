

if __name__ == '__main__':
    detector = FaceDetector()
    faces = detector.buildFaceCoordinates("./images/templates/little_mix_right.jpg")
    print(len(faces))

