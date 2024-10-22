import serial

class Comunicacion_serial:
    def __init__(self):
        try:
            self.com = serial.Serial("COM5", 115200, timeout=1, writeTimeout=10)
        except serial.SerialException as e:
            print(f"Error al conectar con el puerto COM5: {e}")

    def enviar_informacion(self, command: str) -> None:
        try:
            self.com.write(command.encode('ascii'))
        except serial.SerialException as e:
            print(f"Error al enviar información: {e}")

