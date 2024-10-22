import os
import sys
from tkinter import *
import logging as log

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from procesos.main import InterfazGraficaDeUsuario

app= InterfazGraficaDeUsuario(Tk())
app.frame.mainloop()