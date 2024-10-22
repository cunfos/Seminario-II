import numpy as np
from typing import Tuple
from procesos.procesos_de_rostros.herramientos_rostros import HerramientasRostros
from procesos.basededatos.configuracion import RutaBaseDeDatos

class RegistroFacial:
    def __init__(self):
        self.basededatos = RutaBaseDeDatos()
        self.herramientas_faciales = HerramientasRostros()

    def procesos(self, imagen_facial: np.ndarray, dni: str) -> Tuple[np.ndarray, bool, str]:
        # Step 1: check face detection
        checkeo_deteccion_facial, info_facial, guardar_rostro = self.herramientas_faciales.checkeo_facial(imagen_facial)
        if not checkeo_deteccion_facial:
            return imagen_facial, False, 'Rostro no detectado'

        # Step 2: face mesh
        checheo_malla_facial, malla_facial_info = self.herramientas_faciales.malla_facial(imagen_facial)
        if not checheo_malla_facial:
            return imagen_facial, False, 'Malla no detectada'

        # Step 3: extract face mesh
        list_puntos_malla_facial = self.herramientas_faciales.extraer_malla_facial(imagen_facial, malla_facial_info)

        # Step 4: check face center
        checkeo_centro_rostro = self.herramientas_faciales.checkeo_centro_rostro(list_puntos_malla_facial)

        # Step 5: show state
        self.herramientas_faciales.mostrar_estado_registro(imagen_facial, state=checkeo_centro_rostro)

        # Step 6: Only proceed if the face is centered
        if checkeo_centro_rostro:
            # Step 7: extract face info
            bbox_facial = self.herramientas_faciales.extraer_bbox_facial(imagen_facial, info_facial)
            puntos_faciales = self.herramientas_faciales.extraer_puntos_faciales(imagen_facial, info_facial)

            # Step 8: face crop
            recortar_rostro = self.herramientas_faciales.recortar_rostro(guardar_rostro, bbox_facial)

            # Step 9: save face
            checkeo_guardar_imagen = self.herramientas_faciales.guardar_rostro(recortar_rostro, dni, self.basededatos.rostros)
            return imagen_facial, checkeo_guardar_imagen, '¡Imagen guardada!' if checkeo_guardar_imagen else 'Error al guardar la imagen'
        else:
            return imagen_facial, False, 'Por favor, mantenga la cabeza centrada al registrar la imagen'
