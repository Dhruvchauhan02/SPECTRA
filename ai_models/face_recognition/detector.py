# ai_models/face_recognition/detector.py

import cv2
from insightface.app import FaceAnalysis


class RetinaFaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image):
        faces = self.app.get(image)
        return faces
