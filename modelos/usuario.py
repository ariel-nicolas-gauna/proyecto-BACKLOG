import hashlib
import os


class Usuario:
def __init__(self, nombre, correo, edad, rol, password_hash=None, salt=None, label=None):
self.nombre = nombre
self.correo = correo
self.edad = edad
self.rol = rol
self.password_hash = password_hash
self.salt = salt
self.label = label


def ver_perfil(self):
return {
"nombre": self.nombre,
"correo": self.correo,
"edad": self.edad,
"rol": self.rol,
}


def modificar_perfil(self, nombre=None, correo=None, edad=None):
if nombre:
self.nombre = nombre
if correo:
self.correo = correo
if edad:
self.edad = edad
return True


# helpers para contraseña
def make_salt():
return os.urandom(16).hex()


def hash_password(password, salt):
return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
