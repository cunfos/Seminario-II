from pydantic import BaseModel
from procesos.basededatos.ruta_usuarios import (ruta_usuarios, ruta_usuarios_check)
from procesos.basededatos.ruta_rostros import ruta_rostros

class RutaBaseDeDatos(BaseModel):
    # paths
    rostros: str = ruta_rostros
    usuarios: str = ruta_usuarios
    check_usuarios: str = ruta_usuarios_check

