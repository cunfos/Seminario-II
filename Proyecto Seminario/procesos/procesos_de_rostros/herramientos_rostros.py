import os
import numpy as np
import cv2
from ultralytics import YOLO
import datetime
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Tuple, Any
from procesos.procesos_de_rostros.modelo_detector_rostros.detector_rostros import DetectorRostrosMediaPipe
from procesos.procesos_de_rostros.modelo_mallas_rostros.mallas_rostros import MallaFacialMediapipe
from procesos.procesos_de_rostros.modelo_comparacion_rostros.comparacion_rostros import ModeloComparacionRostros
from datetime import datetime

class HerramientasRostros:
    def __init__(self):
        # Detector de rostros
        self.detector_rostros = DetectorRostrosMediaPipe()
        # Malla facial
        self.detector_mallas = MallaFacialMediapipe()
        # Comparador de rostros
        self.comparacion_rostros = ModeloComparacionRostros()
        self.imagen_original_sin_malla = None  # Nueva variable para almacenar la imagen original

        self.angulo = None

        # Variables
        self.rostros_bd = []
        self.nombres_rostros = []
        self.comparador: bool = False
        self.distancia: float = 0.0
        self.usuario_registrado = False

        # Variable para controlar el envío de correos
        self.correo_enviado = False  # Flag booleana para controlar si el correo ya fue enviado

    def checkeo_facial(self, imagen_facial: np.ndarray) -> Tuple[bool, Any, np.ndarray]:
        guardar_rostro = imagen_facial.copy()
        checkeo_facial, info_facial = self.detector_rostros.deteccion_rostros_mediapipe(imagen_facial)
        return checkeo_facial, info_facial, guardar_rostro

    def calcular_ear(self, ojo: np.ndarray) -> float:
        A = np.linalg.norm(ojo[1] - ojo[5])
        B = np.linalg.norm(ojo[2] - ojo[4])
        C = np.linalg.norm(ojo[0] - ojo[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def guardar_imagen_original(self, imagen_sin_malla: np.ndarray):
        """Guarda la imagen original antes de aplicar cualquier malla facial."""
        self.imagen_original_sin_malla = imagen_sin_malla

    def detectar_parpadeo(self, puntos_faciales: List[List[int]], umbral_ear: float) -> bool:
        puntos_ojos_derecho = [33, 160, 158, 133, 153, 144]
        puntos_ojos_izquierdo = [263, 387, 385, 362, 380, 373]
        ojo_derecho = np.array([puntos_faciales[i][1:] for i in puntos_ojos_derecho])
        ojo_izquierdo = np.array([puntos_faciales[i][1:] for i in puntos_ojos_izquierdo])
        ear_derecho = self.calcular_ear(ojo_derecho)
        ear_izquierdo = self.calcular_ear(ojo_izquierdo)
        ear_promedio = (ear_derecho + ear_izquierdo) / 2.0
        return ear_promedio < umbral_ear

    def extraer_bbox_facial(self, imagen_facial: np.ndarray, info_facial: Any):
        h_img, w_img, _ = imagen_facial.shape
        bbox = self.detector_rostros.extraer_face_bbox_mediapipe(w_img, h_img, info_facial)
        return bbox

    def extraer_puntos_faciales(self, imagen_facial: np.ndarray, info_facial: Any):
        h_img, w_img, _ = imagen_facial.shape
        puntos_faciales = self.detector_rostros.extraer_puntos_faciales_mediapipe(h_img, w_img, info_facial)
        return puntos_faciales

    def malla_facial(self, imagen_facial: np.ndarray) -> Tuple[bool, Any]:
        checkeo_malla_facial, info_malla_facial = self.detector_mallas.malla_facial_mediapipe(imagen_facial)
        return checkeo_malla_facial, info_malla_facial

    def extraer_malla_facial(self, imagen_facial: np.ndarray, malla_facial_info: Any) -> List[List[int]]:
        list_puntos_malla_facial = self.detector_mallas.extraer_puntos_malla_facial(imagen_facial, malla_facial_info, viz=True)
        return list_puntos_malla_facial

    def checkeo_centro_rostro(self, puntos_faciales: List[List[int]]) -> bool:
        checkeo_centro_rostro = self.detector_mallas.checkeo_centro_rostro(puntos_faciales)
        return checkeo_centro_rostro

    def recortar_rostro(self, imagen_facial: np.ndarray, bbox_facial: List[int]) -> np.ndarray:
        h, w, _ = imagen_facial.shape
        offset_x, offset_y = int(w * 0.025), int(h * 0.025)
        xi, yi, xf, yf = bbox_facial
        xi, yi, xf, yf = xi - offset_x, yi - (offset_y * 4), xf + offset_x, yf
        return imagen_facial[yi:yf, xi:xf]

    def guardar_rostro(self, recortar_rostro: np.ndarray, dni: str, path: str):
        if len(recortar_rostro) != 0:
            recortar_rostro = cv2.cvtColor(recortar_rostro, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{path}/{dni}.png", recortar_rostro)
            return True
        else:
            return False

    def mostrar_estado_registro(self, imagen_facial: np.ndarray, state: bool):
        if state:
            texto = 'Proceso facial, mire a la camara!'
            color = (0, 255, 0)
        else:
            texto = 'Por favor, mire al frente'
            color = (0, 0, 255)
            self.detector_mallas.config_color((255, 0, 0))
        tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
        dim, baseline = tamaño_texto[0], tamaño_texto[1]
        cv2.rectangle(imagen_facial, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(imagen_facial, texto, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 1)

    def mostrar_estado_login(self, imagen_facial: np.ndarray, state: bool):
        if state:
            texto = 'Acceso concedido. Bienvenido!'
            tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = tamaño_texto[0], tamaño_texto[1]
            cv2.rectangle(imagen_facial, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0), cv2.FILLED)
            cv2.putText(imagen_facial, texto, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)
            self.detector_mallas.config_color((0, 255, 0))
            self.correo_enviado = False  # Resetea el flag si el acceso es concedido

        elif state is None:
            texto = 'Por favor, parpadee 3 veces y luego espere 3 segundos para la verificacion...'
            tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = tamaño_texto[0], tamaño_texto[1]
            cv2.rectangle(imagen_facial, (250, 650 - dim[1] - baseline), (250 + dim[0], 650 + baseline), (0, 0, 0), cv2.FILLED)
            cv2.putText(imagen_facial, texto, (250, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 0), 1)
            self.detector_mallas.config_color((255, 255, 0))

        elif state is False:
            texto = 'Acceso denegado. Registrese para obtener acceso.'
            tamaño_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = tamaño_texto[0], tamaño_texto[1]
            cv2.rectangle(imagen_facial, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0), cv2.FILLED)
            cv2.putText(imagen_facial, texto, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 1), 1)
            self.detector_mallas.config_color((255, 0, 1))

            # Ahora usamos la imagen original sin malla para enviar por correo
            if self.imagen_original_sin_malla is not None:
                imagen_rgb_sin_malla = cv2.cvtColor(self.imagen_original_sin_malla, cv2.COLOR_BGR2RGB)
                nombre_imagen = "rostro_no_autorizado_sin_malla.jpg"
                cv2.imwrite(nombre_imagen, imagen_rgb_sin_malla)

                # Enviar el correo solo si aún no ha sido enviado
                if not self.correo_enviado:
                    threading.Thread(target=self.enviar_correo_alerta, args=(nombre_imagen,)).start()
                    self.correo_enviado = True  # Marca que el correo ya fue enviado

    def enviar_correo_alerta(self, imagen_path: str):
        smtp_server = "smtp.gmail.com"
        port = 587
        sender_email = "pignuoliluca@gmail.com"
        password = "skpl lzyv tsww bhdl"

        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = "pignuoliluca31@gmail.com"
        message["Subject"] = "Intento de acceso no autorizado detectado"

        body = (f"Estimado,\n\n"
                f"Se ha detectado un intento de acceso no autorizado en el sistema de control de acceso.\n"
                f"Adjuntamos la imagen capturada de la persona no autorizada para su referencia.\n\n"
                f"Detalles del intento de acceso:\n"
                f"Fecha y hora: {fecha_hora_actual}\n\n"
                f"Si este acceso no fue autorizado, le recomendamos tomar las medidas de seguridad necesarias.\n\n")

        message.attach(MIMEText(body, "plain"))

        try:
            with open(imagen_path, "rb") as img_file:
                img = MIMEImage(img_file.read())
                img.add_header("Content-Disposition", f"attachment; filename={os.path.basename(imagen_path)}")
                message.attach(img)
        except Exception as e:
            print(f"Error al adjuntar la imagen: {e}")

        try:
            server = smtplib.SMTP(smtp_server, port)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, "pignuoliluca31@gmail.com", message.as_string())
            print("Correo enviado exitosamente con la imagen adjunta")
        except Exception as e:
            print(f"Error al enviar el correo: {e}")
        finally:
            server.quit()

    def leer_rostros_basededatos(self, basededatos_path: str) -> Tuple[List[np.ndarray], List[str], str]:
        self.rostros_bd = []
        self.nombres_rostros = []

        for file in os.listdir(basededatos_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(basededatos_path, file)
                img_read = cv2.imread(img_path)
                if img_read is not None:
                    self.rostros_bd.append(img_read)
                    self.nombres_rostros.append(os.path.splitext(file)[0])

        return self.rostros_bd, self.nombres_rostros, f'Comparando {len(self.rostros_bd)} rostros!'

    def comparador_rostros(self, imagen_actual: np.ndarray, rostros_bd: List[np.ndarray], nombre_bd: List[str]) -> Tuple[bool, str]:
        nombre_usuario = ''
        imagen_actual = cv2.cvtColor(imagen_actual, cv2.COLOR_RGB2BGR)
        for idx, rostro_img in enumerate(rostros_bd):
            self.comparador, self.distancia = self.comparacion_rostros.face_matching_sface_model(imagen_actual, rostro_img)
            print(f'Validando rostro con: {nombre_bd[idx]}')
            if self.comparador:
                nombre_usuario = nombre_bd[idx]
                return True, nombre_usuario

        return False, '¡Rostro desconocido!'

    def usuario_check_in(self, nombre_usuario: str, ruta_usuario: str):
        if not self.usuario_registrado:
            now = datetime.now()
            date_time = now.strftime("%y-%m-%d %H:%M:%S")
            usuario_archivo_path = os.path.join(ruta_usuario, f"{nombre_usuario}.txt")
            with open(usuario_archivo_path, "a") as usuario_archivo:
                usuario_archivo.write(f'\n Acceso garantizado at: {date_time}\n')

            self.usuario_registrado = True
