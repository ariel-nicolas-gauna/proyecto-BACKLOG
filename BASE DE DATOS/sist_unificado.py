import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
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
        print(f"\n--- Perfil de {self.nombre} ---")
        print(f"Correo: {self.correo}")
        print(f"Edad: {self.edad}")
        print(f"Rol: {self.rol}")

    def modificar_perfil(self, nombre=None, correo=None, edad=None):
        if nombre:
            self.nombre = nombre
        if correo:
            self.correo = correo
        if edad:
            self.edad = edad
        print("Perfil actualizado correctamente.")


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

    correo = input("Correo electrónico: ").strip()
    edad = input("Edad: ").strip()
    rol = input("Rol (admin / profesor / estudiante): ").strip().lower()

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
    os.makedirs(user_path, exist_ok=True)

    data["users"][nombre] = {
        "correo": correo,
        "edad": edad,
        "rol": rol,
        "salt": salt,
        "pwd_hash": pwd_hash,
        "label": label
    }
    save_users(data)
    print(f"Usuario {nombre} creado correctamente.")

    resp = input("¿Querés tomar fotos de tu cara ahora para el reconocimiento? (s/n): ").strip().lower()
    if resp == "s":
        capture_faces_for_user(nombre)


def capture_faces_for_user(nombre, cam_index=0, shots=20):
    cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
    print("Presioná 's' para sacar una foto. 'q' para salir.")
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
        cv2.imshow("Captura de caras", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            if len(faces) == 0:
                print("No se detectó cara.")
                continue
            x,y,w,h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            filename = os.path.join(user_path, f"{nombre}_{count}.png")
            cv2.imwrite(filename, face_resized)
            count += 1
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
    print(f"Entrenamiento completo ({len(images)} imágenes).")


def login(cam_index=0, recognizer_path=RECOGNIZER_FILE, confidence_threshold=60):
    data = load_users()
    if not os.path.exists(recognizer_path):
        print("No hay modelo entrenado.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(recognizer_path)

    cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("No se puede acceder a la cámara.")
        return None

    print("Intentando reconocimiento facial... (presioná 'q' para cancelar)")
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
            for user, info in data["users"].items():
                if info["label"] == label:
                    user_name = user
                    break
            else:
                user_name = None

            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, f"{user_name or 'Desconocido'} ({int(conf)})", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

            if user_name and conf <= confidence_threshold:
                recognized_user = user_name
                print(f"Reconocido: {recognized_user} (conf {conf:.1f})")
                cap.release()
                cv2.destroyAllWindows()
                return recognized_user

        cv2.imshow("Login facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    nombre = input("Usuario: ").strip()
    if nombre not in data["users"]:
        print("Usuario no existe.")
        return None
    pwd = safe_getpass("Contraseña: ")
    info = data["users"][nombre]
    if hash_password(pwd, info["salt"]) == info["pwd_hash"]:
        print("Login correcto.")
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
    doblados = all(distancia(landmarks[i], landmarks[0]) < distancia(landmarks[i - 2], landmarks[0])
                   for i in dedos_tips)
    mano_recta = indice_mcp.x < pulgar_mcp.x < menique_mcp.x or menique_mcp.x < pulgar_mcp.x < indice_mcp.x
    return apunta_arriba and doblados and mano_recta


def sesion_usuario(nombre, cam_index=0):
    data = load_users()
    info = data["users"][nombre]
    rol = info["rol"]
    print(f"\nBienvenido {nombre} ({rol})")

    if rol == "admin":
        print("\nUsuarios registrados:")
        for u, i in data["users"].items():
            print(f"- {u} ({i['rol']})")
    elif rol == "estudiante":
        print("\nHacé el gesto de pulgar arriba 👍 para confirmar asistencia.")
        cap = cv2.VideoCapture(cam_index)
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            confirmado = False
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                # MediaPipe expects RGB input, OpenCV uses BGR for display.
                # Convert for processing, then draw on a BGR copy for correct colors in imshow.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                display_frame = frame.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # draw landmarks on BGR image for correct color rendering
                        mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        if detectar_pulgar_arriba(hand_landmarks.landmark):
                            cv2.putText(display_frame, "Gesto correcto 👍", (30,50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),3)
                            confirmado = True
                if confirmado:
                    cv2.putText(display_frame, f"{nombre} confirmado como estudiante", (30,100),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                cv2.imshow("Confirmación de pulgar", display_frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Sesión iniciada correctamente.")


def main_menu():
    print("Sistema completo - Login por rostro, contraseña y gesto\n")
    while True:
        print("\nOpciones:")
        print("1) Registrar usuario")
        print("2) Capturar fotos para usuario existente")
        print("3) Entrenar reconocedor")
        print("4) Iniciar sesión (cara o contraseña)")
        print("5) Salir")
        opt = input("Elegí una opción (1-5): ").strip()
        if opt == "1":
            register_user()
        elif opt == "2":
            u = input("Usuario: ").strip()
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


def run_in_thread(target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()


class App:
    def __init__(self, root):
        self.root = root
        root.title('Sistema - GUI')

        frm = tk.Frame(root, padx=10, pady=10)
        frm.pack()

        tk.Label(frm, text='Índice de cámara:').grid(row=0, column=0, sticky='w')
        self.cam_idx_var = tk.StringVar(value='0')
        tk.Entry(frm, textvariable=self.cam_idx_var, width=5).grid(row=0, column=1, sticky='w')

        
        tk.Button(frm, text='1) Registrar usuario', width=30, command=self.on_register).grid(row=1, column=0, columnspan=2, pady=4)

       
        tk.Label(frm, text='Usuario para captura:').grid(row=2, column=0, sticky='w')
        self.capture_user_var = tk.StringVar()
        tk.Entry(frm, textvariable=self.capture_user_var).grid(row=2, column=1, sticky='w')
        tk.Button(frm, text='2) Capturar fotos', width=30, command=self.on_capture).grid(row=3, column=0, columnspan=2, pady=4)

       
        tk.Button(frm, text='3) Entrenar reconocedor', width=30, command=self.on_train).grid(row=4, column=0, columnspan=2, pady=4)

        
        tk.Button(frm, text='4) Iniciar sesión (cara / contraseña)', width=30, command=self.on_login).grid(row=5, column=0, columnspan=2, pady=4)

        
        tk.Button(frm, text='5) Salir', width=30, command=root.quit).grid(row=6, column=0, columnspan=2, pady=4)

        
    def get_cam_idx(self):
        try:
            return int(self.cam_idx_var.get())
        except Exception:
            return 0

    def on_register(self):
      
        if messagebox.askyesno('Registrar', 'Registrar usuario por consola (sí) o cancelar (no)?'):
            run_in_thread(register_user)

    def on_capture(self):
        user = self.capture_user_var.get().strip()
        if not user:
            messagebox.showinfo('Falta usuario', 'Ingresá el nombre de usuario en "Usuario para captura"')
            return
        cam = self.get_cam_idx()
        run_in_thread(capture_faces_for_user, user, cam)
        messagebox.showinfo('Captura iniciada', f'Se abrió la ventana de captura para {user} (índice {cam})')

    def on_train(self):
        if messagebox.askyesno('Entrenar', 'Entrenar el reconocedor con las imágenes en users/?'):
            run_in_thread(train_recognizer)
            messagebox.showinfo('Entrenamiento', 'Entrenamiento iniciado en segundo plano. Revisá la consola.')

    def on_login(self):
        cam = self.get_cam_idx()

        def attempt():
            user = login(cam_index=cam)
            if user:
                messagebox.showinfo('Login', f'Usuario reconocido: {user}')
                # abrir sesión (usa OpenCV/MediaPipe)
                sesion_usuario(user, cam_index=cam)
                return

            # fallback GUI para contraseña
            users = load_users()
            username = simpledialog.askstring('Login por contraseña', 'Usuario:')
            if not username:
                return
            if username not in users['users']:
                messagebox.showerror('Error', 'Usuario no existe')
                return
            pwd = simpledialog.askstring('Contraseña', 'Contraseña:', show='*')
            if not pwd:
                return
            info = users['users'][username]
            if hash_password(pwd, info['salt']) == info['pwd_hash']:
                messagebox.showinfo('Login', 'Login correcto')
                sesion_usuario(username, cam_index=cam)
            else:
                messagebox.showerror('Login', 'Contraseña incorrecta')

        run_in_thread(attempt)


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
