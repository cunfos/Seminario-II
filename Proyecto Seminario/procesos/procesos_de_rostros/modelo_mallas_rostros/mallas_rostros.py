import numpy as np
import mediapipe as mp
import cv2
from typing import Any, List, Tuple


class MallaFacialMediapipe:
    def __init__(self):
        # Inicialización de Mediapipe y configuración de dibujo
        self.dibujo_mp = mp.solutions.drawing_utils
        # Configuración para dibujar puntos faciales (color gris claro, grosor reducido)
        self.config_dibujo = self.dibujo_mp.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1)

        # Configuración de la detección de malla facial
        self.objeto_malla_facial = mp.solutions.face_mesh
        self.malla_facial_mp = self.objeto_malla_facial.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.puntos_malla = None

        # Coordenadas de puntos específicos de la cara
        # Parietal derecho
        self.rp_x: int = 0
        self.rp_y: int = 0
        # Parietal izquierdo
        self.lp_x: int = 0
        self.lp_y: int = 0
        # Cejas derecha
        self.re_x: int = 0
        self.re_y: int = 0
        # Cejas izquierda
        self.le_x: int = 0
        self.le_y: int = 0

    def malla_facial_mediapipe(self, imagen_facial: np.ndarray) -> Tuple[bool, Any]:
        """
        Procesa la imagen para detectar la malla facial usando Mediapipe.
        Convierte la imagen a RGB antes de realizar el análisis.
        """
        imagen_rgb = cv2.cvtColor(imagen_facial.copy(), cv2.COLOR_BGR2RGB)
        malla_facial = self.malla_facial_mp.process(imagen_rgb)

        # Verificar si se detectó la malla facial
        if malla_facial.multi_face_landmarks is None:
            return False, malla_facial
        else:
            return True, malla_facial

    def extraer_puntos_malla_facial(self, imagen_facial: np.ndarray, malla_facial_info: Any, viz: bool) -> List[
        List[int]]:
        """
        Extrae las coordenadas de los puntos faciales y, si se solicita, visualiza la malla en la imagen.
        """
        height, width, _ = imagen_facial.shape
        self.puntos_malla = []

        # Recorre todos los puntos detectados para obtener las coordenadas
        for malla_facial in malla_facial_info.multi_face_landmarks:
            for i, puntos in enumerate(malla_facial.landmark):
                x, y = int(puntos.x * width), int(puntos.y * height)
                self.puntos_malla.append([i, x, y])

            # Visualización opcional de los puntos faciales sin líneas
            if viz:
                self.dibujo_mp.draw_landmarks(
                    image=imagen_facial,
                    landmark_list=malla_facial,
                    connections=None,  # No dibujar conexiones para que solo se vean los puntos
                    landmark_drawing_spec=self.config_dibujo
                )

        return self.puntos_malla

    def checkeo_centro_rostro(self, puntos_faciales: List[List[int]]) -> bool:
        """
        Verifica la posición del centro del rostro basándose en las coordenadas de puntos clave.
        """
        if len(puntos_faciales) == 468:
            # Actualiza las coordenadas de los puntos clave
            self.rp_x, self.rp_y = puntos_faciales[139][1:3]  # Parietal derecho
            self.lp_x, self.lp_y = puntos_faciales[368][1:3]  # Parietal izquierdo
            self.re_x, self.re_y = puntos_faciales[70][1:3]  # Cejas derecha
            self.le_x, self.le_y = puntos_faciales[300][1:3]  # Cejas izquierda

            # Comprobación del centro del rostro
            return self.re_x > self.rp_x and self.le_x < self.lp_x

    def config_color(self, color: Tuple[int, int, int]):
        """
        Configura el color de los puntos en la malla facial.
        """
        self.config_dibujo = self.dibujo_mp.DrawingSpec(color=color, thickness=1, circle_radius=1)
