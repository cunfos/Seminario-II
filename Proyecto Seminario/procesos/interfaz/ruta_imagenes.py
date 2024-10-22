from pydantic import BaseModel

from procesos.interfaz.setup.imagenes.face_capture_button import face_capture_image_path
from procesos.interfaz.setup.imagenes.gui_init_image import gui_init_image_path
from procesos.interfaz.setup.imagenes.gui_signup_image import gui_signup_image_path
from procesos.interfaz.setup.imagenes.login_button import login_button_image_path
from procesos.interfaz.setup.imagenes.signup_button import signup_button_image_path


class RutaImagenes(BaseModel):
    # main images
    init_img: str = gui_init_image_path
    login_img: str = login_button_image_path
    signup_img: str = signup_button_image_path

    # secondary windows
    gui_signup_img: str = gui_signup_image_path
    register_img: str = face_capture_image_path
