import numpy as np
from typing import Tuple
from procesos.procesos_de_rostros.herramientos_rostros import HerramientasRostros
from procesos.basededatos.configuracion import RutaBaseDeDatos
import cv2

class FacialLogin:
    def __init__(self):
        self.basededatos = RutaBaseDeDatos()
        self.herramientas_faciales = HerramientasRostros()

        self.comparador = None
        self.comparacion = False
        self.cont_frame = 0
        self.contador_parpadeos = 0
        self.estado_ojo = 'abierto'

        # Umbral EAR para considerar un parpadeo (ajusta según necesidad)
        self.UMBRAL_EAR = 0.2
        # Se requiere parpadear 3 veces
        self.PARPAD_UMBRAL = 3

    def procesos(self, imagen_facial: np.ndarray):
        # Capturar la imagen original antes de cualquier procesamiento
        guardar_rostro_original = imagen_facial.copy()

        # step 1: check face detection
        checkeo_deteccion_facial, info_facial, guardar_rostro = self.herramientas_faciales.checkeo_facial(imagen_facial)
        if not checkeo_deteccion_facial:
            return imagen_facial, self.comparador, 'Rostro no detectado'

        # step 2: face mesh
        checkeo_malla_facial, malla_facial_info = self.herramientas_faciales.malla_facial(imagen_facial)
        if not checkeo_malla_facial:
            return imagen_facial, self.comparador, 'Malla no detectada'

        # Aquí guardamos la imagen sin la malla antes de continuar
        self.herramientas_faciales.guardar_imagen_original(guardar_rostro_original)

        # Continúa con el resto del procesamiento como antes...

        # step 3: extract face mesh
        list_puntos_malla_facial = self.herramientas_faciales.extraer_malla_facial(imagen_facial, malla_facial_info)

        # step 4: check face center
        checkeo_centro_rostro = self.herramientas_faciales.checkeo_centro_rostro(list_puntos_malla_facial)

        # Mostrar el estado de registro si el rostro no está centrado (función original)
        if not checkeo_centro_rostro:
            self.herramientas_faciales.mostrar_estado_registro(imagen_facial, state=False)
            return imagen_facial, self.comparador, 'Por favor, mire al frente'

        # Mostrar el estado de login si el rostro está centrado (función original)
        if checkeo_centro_rostro:
            self.herramientas_faciales.mostrar_estado_login(imagen_facial, state=self.comparador)

        # step 5: Detectar parpadeo usando la función en HerramientasRostros
        if self.herramientas_faciales.detectar_parpadeo(list_puntos_malla_facial, self.UMBRAL_EAR):
            if self.estado_ojo == 'abierto':
                self.contador_parpadeos += 1
                self.estado_ojo = 'cerrado'  # Cambiamos el estado del ojo
        else:
            self.estado_ojo = 'abierto'  # El ojo ya no está cerrado

        # Mostrar el contador de parpadeos en la pantalla
        texto_parpadeos = f'Parpadeos: {self.contador_parpadeos} / {self.PARPAD_UMBRAL}'
        cv2.putText(imagen_facial, texto_parpadeos, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

        # Solo si el usuario ha parpadeado 3 veces, continuar con el proceso de login
        if self.contador_parpadeos >= self.PARPAD_UMBRAL:
            self.cont_frame += 1
            if self.cont_frame == 48:  # Espera unos cuantos frames antes de proceder
                # step 7: extraer bbox y puntos faciales
                bbox_facial = self.herramientas_faciales.extraer_bbox_facial(imagen_facial, info_facial)
                puntos_faciales = self.herramientas_faciales.extraer_puntos_faciales(imagen_facial, info_facial)

                # step 8: face crop
                recortar_rostro = self.herramientas_faciales.recortar_rostro(guardar_rostro, bbox_facial)

                # step 9: leer base de datos de rostros
                basededatos_rostros, basededatos_nombres, info = self.herramientas_faciales.leer_rostros_basededatos(self.basededatos.rostros)

                if len(basededatos_rostros) != 0 and not self.comparacion and self.comparador is None:
                    self.comparacion = True
                    # step 10: comparar rostros
                    self.comparador, nombre_usuario = self.herramientas_faciales.comparador_rostros(recortar_rostro, basededatos_rostros, basededatos_nombres)

                    if self.comparador:
                        # step 11: acceso aprobado, guardar información
                        self.herramientas_faciales.usuario_check_in(nombre_usuario, self.basededatos.usuarios)
                        return imagen_facial, self.comparador, 'Acceso a usuario aprobado!'
                    else:
                        return imagen_facial, self.comparador, 'Usuario no aprobado'
                else:
                    return imagen_facial, self.comparador, 'Base de datos vacía'
            else:
                return imagen_facial, self.comparador, 'Espera un momento...'
        else:
            return imagen_facial, self.comparador, f'Por favor, parpadee {self.PARPAD_UMBRAL - self.contador_parpadeos} veces más para continuar.'
