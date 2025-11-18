import os
import cv2
import json
import time
import math
import hashlib
import numpy as np
import sys
import mediapipe as mp

try:
    from getpass import getpass as _getpass
except Exception:
    _getpass = None

USERS_DIR = "users"
USERS_FILE = "users.json"
RECOGNIZER_FILE = "recognizer.yml"
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(USERS_DIR, exist_ok=True)
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({"next_label": 0, "users": {}}, f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def make_salt():
    return os.urandom(16).hex()

def hash_password(password, salt):
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def tk_getpass(prompt="Contraseña: "):
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        pwd = simpledialog.askstring("Contraseña", prompt, show='*', parent=root)
        root.destroy()
        return "" if pwd is None else pwd
    except Exception:
        return ""

def safe_getpass(prompt="Contraseña: "):
    try:
        if _getpass is not None:
            try:
                p = _getpass(prompt)
                if p is not None:
                    return p
            except Exception:
                pass
    except NameError:
        pass
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                return input(prompt)
            except Exception:
                pass
    except Exception:
        pass
    return tk_getpass(prompt)

def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        data = {"next_label": 0, "users": {}}
        with open(USERS_FILE, "w") as f:
            json.dump(data, f)
        return data

def save_users(data):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Error guardando users.json:", e)

def register_user():
    try:
        data = load_users()

        nombre = input("Nombre de usuario (sin espacios): ").strip()
        if not nombre:
            print("Nombre vacío. Cancelado.")
            return
        if " " in nombre:
            print("El nombre no puede contener espacios. Cancelado.")
            return
        if nombre in data["users"]:
            print("El usuario ya existe.")
            return

        rol = input("Rol (ej. estudiante): ").strip().lower()
        if not rol:
            print("Rol vacío. Cancelado.")
            return

        pwd1 = safe_getpass("Contraseña (se oculta si es posible): ")
        if not pwd1:
            print("Contraseña vacía o cancelada. Cancelado.")
            return
        pwd2 = safe_getpass("Confirmar contraseña: ")
        if pwd1 != pwd2:
            print("Las contraseñas no coinciden. Cancelado.")
            return

        salt = make_salt()
        pwd_hash = hash_password(pwd1, salt)
        label = data["next_label"]
        data["next_label"] += 1

        user_path = os.path.join(USERS_DIR, nombre)
        try:
            os.makedirs(user_path, exist_ok=True)
        except Exception as e:
            print("No se pudo crear la carpeta del usuario:", e)
            return

        data["users"][nombre] = {
            "rol": rol,
            "salt": salt,
            "pwd_hash": pwd_hash,
            "label": label
        }
        save_users(data)
        print(f"Usuario {nombre} creado con label {label}.")

        resp = input("¿Querés tomar fotos de tu cara ahora para el reconocimiento? (s/n): ").strip().lower()
        if resp == "s":
            capture_faces_for_user(nombre)
    except Exception as e:
        print("Ocurrió un error durante el registro:", e)

def capture_faces_for_user(nombre, cam_index=0, shots=30):
    cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Verificá DroidCam o el índice de cámara.")
        return
    print("Presioná 's' para sacar una foto (se necesitan varias). 'q' para salir.")
    count = 0
    user_path = os.path.join(USERS_DIR, nombre)
    os.makedirs(user_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)

        cv2.putText(frame, f"Fotos tomadas: {count}/{shots}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.imshow("Captura de caras - presionar s para salvar", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            if len(faces) == 0:
                print("No se detectó cara. Acercate/frente a la cámara.")
                continue
            x,y,w,h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            filename = os.path.join(user_path, f"{nombre}_{count}.png")
            cv2.imwrite(filename, face_resized)
            count += 1
            print(f"Guardada {filename}")
            if count >= shots:
                print("Tomas completadas.")
                break
        elif k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_recognizer(save_path=RECOGNIZER_FILE):
    cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    data = load_users()
    images = []
    labels = []
    for user, info in data["users"].items():
        label = info["label"]
        user_path = os.path.join(USERS_DIR, user)
        if not os.path.isdir(user_path):
            continue
        for fn in os.listdir(user_path):
            if fn.lower().endswith((".png",".jpg",".jpeg")):
                img = cv2.imread(os.path.join(user_path, fn), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_resized = cv2.resize(img, (200,200))
                images.append(img_resized)
                labels.append(label)
    if len(images) == 0:
        print("No hay imágenes para entrenar.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.write(save_path)
    print(f"Reconocedor entrenado y guardado en {save_path} con {len(images)} imágenes.")

def login(cam_index=0, recognizer_path=RECOGNIZER_FILE, confidence_threshold=60):
    data = load_users()
    if not os.path.exists(recognizer_path):
        print("No hay modelo entrenado. Ejecutá 'train' primero o registrá y entrená.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(recognizer_path)

    cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("No se puede abrir la cámara. Revisá DroidCam.")
        return None

    print("Intentando reconocimiento facial. Mirá la cámara. Presioná 'q' para cancelar.")
    start = time.time()
    recognized_user = None

    while time.time() - start < 10:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200,200))
            label, conf = recognizer.predict(face_resized)
            user_name = None
            for user, info in data["users"].items():
                if info["label"] == label:
                    user_name = user
                    break

            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, f"{user_name or 'Desconocido'} ({int(conf)})", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

            if user_name is not None and conf <= confidence_threshold:
                recognized_user = user_name
                print(f"Reconocido: {recognized_user} (conf {conf:.1f})")
                cap.release()
                cv2.destroyAllWindows()
                return recognized_user

        cv2.imshow("Login por cara (presionar q para cancelar)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("No se reconoció la cara. Podés loguearte con usuario + contraseña.")
    nombre = input("Usuario: ").strip()
    if nombre not in data["users"]:
        print("Usuario no existe.")
        return None
    pwd = safe_getpass("Contraseña: ")
    info = data["users"][nombre]
    if hash_password(pwd, info["salt"]) == info["pwd_hash"]:
        print("Contraseña correcta. Logueado.")
        return nombre
    else:
        print("Contraseña incorrecta.")
        return None

def distancia(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def detectar_pulgar_arriba(landmarks):
    pulgar_tip = landmarks[4]
    pulgar_mcp = landmarks[2]
    indice_mcp = landmarks[5]
    menique_mcp = landmarks[17]

    dx = pulgar_tip.x - pulgar_mcp.x
    dy = pulgar_tip.y - pulgar_mcp.y
    angulo = math.degrees(math.atan2(dy, dx))
    apunta_arriba = -160 < angulo < -20

    dedos_tips = [8, 12, 16, 20]
    doblados = all(
        distancia(landmarks[i], landmarks[0]) < distancia(landmarks[i - 2], landmarks[0])
        for i in dedos_tips
    )

    mano_recta = indice_mcp.x < pulgar_mcp.x < menique_mcp.x or menique_mcp.x < pulgar_mcp.x < indice_mcp.x
    return apunta_arriba and doblados and mano_recta

def sesion_usuario(nombre, cam_index=0):
    data = load_users()
    if nombre not in data["users"]:
        print("Error: usuario no encontrado.")
        return
    rol = data["users"][nombre]["rol"]
    print(f"Bienvenido {nombre}. Rol: {rol}")

    if rol != "estudiante":
        print("Solo los estudiantes necesitan confirmar con gesto en este flujo.")
        return

    print("Comienza detección del gesto. Hacé pulgar arriba 👍 para confirmar.")
    cap = cv2.VideoCapture(cam_index)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        confirmado = False
        ultimo_gesto = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if detectar_pulgar_arriba(hand_landmarks.landmark):
                        cv2.putText(frame, "Gesto correcto 👍", (30,50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),3)
                        confirmado = True
                        ultimo_gesto = time.time()
            if confirmado and time.time() - ultimo_gesto < 2:
                cv2.rectangle(frame, (20,20),(620,120),(0,255,0),3)
                cv2.putText(frame, f"{nombre} confirmado como estudiante", (30,100),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            cv2.imshow("Confirmación de pulgar - presionar ESC para salir", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

def main_menu():
    print("Sistema de login - cara o contraseña\n")
    while True:
        print("\nOpciones:")
        print("1) Registrar usuario")
        print("2) Capturar fotos para usuario existente")
        print("3) Entrenar reconocedor")
        print("4) Login (cara o contraseña)")
        print("5) Salir")
        opt = input("Elegí una opción (1-5): ").strip()
        if opt == "1":
            register_user()
        elif opt == "2":
            u = input("Usuario para capturar fotos: ").strip()
            if u in load_users()["users"]:
                capture_faces_for_user(u)
            else:
                print("Usuario no existe.")
        elif opt == "3":
            train_recognizer()
        elif opt == "4":
            cam_idx = input("Índice de cámara (enter para 0): ").strip()
            cam_idx = int(cam_idx) if cam_idx != "" else 0
            user = login(cam_index=cam_idx)
            if user:
                sesion_usuario(user, cam_index=cam_idx)
        elif opt == "5":
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main_menu()
