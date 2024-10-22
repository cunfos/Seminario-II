import numpy as np
import mediapipe as mp
import cv2
from typing import Any, Tuple


class DetectorRostrosMediaPipe:
    def __init__(self):
        #mediapipe
        self.objeto_rostro_mp = mp.solutions.face_detection
        self.detector_facial_mp = self.objeto_rostro_mp.FaceDetection(min_detection_confidence=0.7, model_selection=0)
        self.bbox = []
        self.puntos_faciales = []

    def deteccion_rostros_mediapipe(self, imagen_rostro: np.ndarray)-> Tuple[bool, Any]:
       imagen_rgb = imagen_rostro.copy()
       imagen_rgb = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2RGB)

       rostros = self.detector_facial_mp.process(imagen_rgb)
       if rostros.detections is None:
           return False, rostros
       else:
           return True, rostros

    def extraer_face_bbox_mediapipe(self, width_img: int, height_img: int, face_info: Any):
        self.bbox = []
        for face in face_info.detections:
            bbox = face.location_data.relative_bounding_box
            xi, yi, w_face, h_face = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            xi, yi, w_face, h_face = int(xi * width_img), int(yi * height_img), int(w_face * width_img), int(h_face * height_img)
            xf, yf = xi + w_face, yi + h_face

            xi = max(0, xi)
            yi = max(0, yi)
            xf = min(width_img, xf)
            yf = min(height_img, yf)

            self.bbox = [xi, yi, xf, yf]

        return self.bbox


    def extraer_puntos_faciales_mediapipe(self, ancho_img: int, largo_img: int, info_facial: Any):
        self.puntos_faciales = []
        for rostro in info_facial.detections:
                key_points = rostro.location_data.relative_keypoints
                for i, points in enumerate(key_points):
                    x, y = int(points.x * ancho_img), int(points.y * largo_img)
                    self.puntos_faciales.append([x, y])
        return self.puntos_faciales
