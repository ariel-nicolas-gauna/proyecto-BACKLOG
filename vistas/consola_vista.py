class ConsolaView:
def menu(self, controller):
while True:
print("\n--- MENÚ CONSOLA ---")
print("1. Registrar usuario")
print("2. Iniciar sesión")
print("3. Salir")
op = input("> ")


if op == "1":
n = input("Nombre: ")
c = input("Correo: ")
e = int(input("Edad: "))
p = input("Password: ")
print(controller.registrar(n, c, e, p))


elif op == "2":
c = input("Correo: ")
p = input("Password: ")
print(controller.login(c, p))


elif op == "3":
print("Saliendo...")
break


else:
print("Opción inválida")
