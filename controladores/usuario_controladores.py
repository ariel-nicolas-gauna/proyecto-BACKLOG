from usuario_model import UsuarioModel


class UsuarioController:
def __init__(self):
self.model = UsuarioModel()


def registrar(self, nombre, correo, edad, password):
if self.model.obtener_por_correo(correo):
return "Error: el usuario ya existe"
self.model.crear_usuario(nombre, correo, edad, password)
return "Usuario registrado con éxito"


def login(self, correo, password):
u = self.model.obtener_por_correo(correo)
if u and u.password == password:
return f"Bienvenido {u.nombre}"
return "Credenciales incorrectas"
