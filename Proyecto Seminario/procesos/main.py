import os
from tkinter import *
import tkinter as Tk
import imutils
from PIL import Image, ImageTk
import cv2
from scipy.cluster.hierarchy import weighted
from procesos.interfaz.ruta_imagenes import RutaImagenes
from procesos.basededatos.configuracion import RutaBaseDeDatos
from procesos.interfaz.setup.imagenes.login_button import login_button_image_path
from procesos.procesos_de_rostros.registro_rostro import RegistroFacial
from procesos.procesos_de_rostros.logeo_rostro import FacialLogin
from procesos.interfaz_comunicacion.comunicacion_serial import Comunicacion_serial

class CustomFrame(Tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(fill=Tk.BOTH, expand=True)

class InterfazGraficaDeUsuario:
    def __init__(self, root):
        self.main_window = root
        self.main_window.title('Control de acceso facial')
        self.main_window.geometry('1280x720')
        self.frame = CustomFrame(self.main_window)

        #Video en directo
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        #singup windows
        self.registro_window = None
        self.entrada_nombre = None
        self.entrada_codigo_usuario = None
        self.nombre = None
        self.dni = None
        self.lista_usuario = None

        # face capture
        self.registro_facial_window = None
        self.registro_video = None
        self.dni_usuario = []
        self.datos = []

        #login windows
        self.login_facial_window = None
        self.login_video = None

        # modulos
        self.images= RutaImagenes()
        self.basededatos = RutaBaseDeDatos()
        self.registro_facial = RegistroFacial()
        self.facial_login = FacialLogin()
        self.com = Comunicacion_serial()

        #procesos
        self.main()

    def cerrar_login(self):
        self.facial_login.__init__()
        self.login_facial_window.destroy()
        self.login_video.destroy()
        self.com.enviar_informacion('C')
    def login_facial(self):
        if self.cap:
            ret, frame_bgr = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # proceso
                frame, acceso_usuarios,info = self.facial_login.procesos(frame)

                # config video
                frame = imutils.resize(frame, width=1280)
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)

                # show frame
                self.login_video.configure(image=img)
                self.login_video.image = img
                self.login_video.after(10, self.login_facial)

                if acceso_usuarios:
                    # Abrir la puerta enviando comando 'O'
                    self.com.enviar_informacion('O')
                    self.login_video.after(3000, self.cerrar_login)  # Mantener la puerta abierta por 3 segundos
                elif acceso_usuarios is False:
                    # Cerrar la puerta enviando comando 'C'
                    self.com.enviar_informacion('C')
                    self.login_video.after(3000, self.cerrar_login)
        else:
            self.cap.release()

    def interfaz_login(self):

        self.login_facial_window = Toplevel()
        self.login_facial_window.title('Inicio de sesion facial')
        self.login_facial_window.geometry('1280x720')

        self.login_video = Label(self.login_facial_window)
        self.login_video.place(x=0, y=0)
        self.login_facial()

    def cerrar_registro(self):
        self.registro_facial.__init__()
        self.registro_facial_window.destroy()
        self.registro_video.destroy()

    def registro_rostros(self):
        if self.cap:
            ret, frame_bgr = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                #proceso
                frame, imagen_guardada, info = self.registro_facial.procesos(frame, self.dni)

                #config video
                frame = imutils.resize(frame, width=1280)
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)

                #show frame
                self.registro_video.configure(image=img)
                self.registro_video.image = img
                self.registro_video.after(10, self.registro_rostros)

                if imagen_guardada:
                    self.registro_video.after(3000, self.cerrar_registro)
        else:
            self.cap.release()

    def registro_datos(self):
        #leer informacion
        self.nombre, self.dni = self.entrada_nombre.get(), self.entrada_codigo_usuario.get()
        # check data
        if len(self.nombre) == 0 or len(self.dni)== 0:
            print('Formulario Incompleto!')
        else:
            #check user
            self.lista_usuario = os.listdir(self.basededatos.check_usuarios)
            for u_list in self.lista_usuario:
                usuario = u_list
                usuario = usuario.split('.')
                self.dni_usuario.append(usuario[0])
            if self.dni in self.dni_usuario:
                print('Usuario ya registrado')
            else:
                #save data
                self.datos.append(self.nombre)
                self.datos.append(self.dni_usuario)

                file = open(f"{self.basededatos.usuarios}/{self.dni}.txt", 'w')
                file.writelines(self.nombre + ',')
                file.writelines(self.dni + ',')
                file.close()

                #clear
                self.entrada_nombre.delete(0, END)
                self.entrada_codigo_usuario.delete(0, END)

                #face register
                self.registro_facial_window = Toplevel()
                self.registro_facial_window.title('Captura Facial')
                self.registro_facial_window.geometry('1280x720')

                self.registro_video = Label(self.registro_facial_window)
                self.registro_video.place(x=0, y=0)
                self.registro_window.destroy()
                self.registro_rostros()

    def interfaz_registro(self):
        self.registro_window = Toplevel(self.frame)
        self.registro_window.title('Registro Facial')
        self.registro_window.geometry("1280x720")

        # background
        fondo_registro_img = PhotoImage(file=self.images.gui_signup_img)
        fondo_registro = Label(self.registro_window, image=fondo_registro_img)
        fondo_registro.img = fondo_registro_img
        fondo_registro.place(x=0, y=0)

        #input data
        self.entrada_nombre = Entry(self.registro_window)
        self.entrada_nombre.place(x=575, y=260)
        self.entrada_codigo_usuario = Entry(self.registro_window)
        self.entrada_codigo_usuario.place(x=573, y=432)

        #input button
        boton_registro_img = PhotoImage(file=self.images.register_img)
        boton_registro = Button(self.registro_window, image=boton_registro_img, height=40, width=200, command=self.registro_datos)
        boton_registro.image = boton_registro_img
        boton_registro.place(x=1005, y=564)

    def main(self):
        # background
        background_img = PhotoImage(file=self.images.init_img)
        background = Label(self.frame, image=background_img, text='back')
        background.image = background_img
        background.place(x=0, y=0, relwidth=1, relheight=1)

        # botones
        loging_button_img = PhotoImage(file=self.images.login_img)
        login_button = Button(self.frame, image=loging_button_img, height=40, width=200, command=self.interfaz_login)
        login_button.img = loging_button_img
        login_button.place(x=980, y=325)

        singup_button_img = PhotoImage(file=self.images.signup_img)
        singup_button = Button(self.frame, image=singup_button_img, height=40, width=200, command=self.interfaz_registro)
        singup_button.img = singup_button_img
        singup_button.place(x=980, y=582)
