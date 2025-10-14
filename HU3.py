def cargar_usuario():
    nombre = input("Ingrese su nombre: ")
    tipo = input("¿Eres estudiante o profesor? (E/P): ").strip().upper()
    return nombre, tipo

def reconocer_gesto(tipo_usuario):
    if tipo_usuario == 'E':
        print("Gesto reconocido: Estudiante - mueve el pulgar para arriba 👍")
    elif tipo_usuario == 'P':
        print("Gesto reconocido: Profesor - mueve el pulgar para abajo 👎")
    else:
        print("Tipo de usuario no reconocido.")

def main():
    nombre, tipo = cargar_usuario()
    reconocer_gesto(tipo)

if __name__ == "__main__":
    main()
