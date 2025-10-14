class Usuario:
    def __init__(self, nombre, correo, edad):
        self.nombre = nombre
        self.correo = correo
        self.edad = edad

    def ver_perfil(self):
        print(f"Nombre: {self.nombre}")
        print(f"Correo: {self.correo}")
        print(f"Edad: {self.edad}")

    def modificar_perfil(self, nombre=None, correo=None, edad=None):
        if nombre:
            self.nombre = nombre
        if correo:
            self.correo = correo
        if edad:
            self.edad = edad
        print("Perfil actualizado correctamente.")

# Ejemplo de uso:
usuario = Usuario("Juan", "juan@email.com", 25)
usuario.ver_perfil()
usuario.modificar_perfil(nombre="Juan Pérez", edad=26)
usuario.ver_perfil()
