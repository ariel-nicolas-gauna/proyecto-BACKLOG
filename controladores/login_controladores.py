import time
from models.almacenamiento import load_users
from models.reconocimiento import load_recognizer
from models.usuario import hash_password




def attempt_facial_login(cam_index=0, recognizer_path=None, confidence_threshold=60, timeout=10):
data = load_users()
recognizer = load_recognizer(recognizer_path) if recognizer_path else load_recognizer()
if recognizer is None:
print("No hay modelo entrenado.")
return None


cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
print("No se puede acceder a la cámara.")
return None


print("Intentando reconocimiento facial... (presioná 'q' para cancelar)")
start = time.time()


while time.time() - start < timeout:
ret, frame = cap.read()
if not ret:
break
frame = cv2.flip(frame, 1)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
face = gray[y:y + h, x:x + w]
face_resized = cv2.resize(face, (200, 200))
label, conf = recognizer.predict(face_resized)
user_name = None
for user, info in data["users"].items():
if info["label"] == label:
user_name = user
break


cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(frame, f"{user_name or 'Desconocido'} ({int(conf)})", (x, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


if user_name and conf <= confidence_threshold:
cap.release()
cv2.destroyAllWindows()
return user_name


cv2.imshow("Login facial", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break


cap.release()
cv2.destroyAllWindows()
return None




def password_login_console():
users = load_users()
nombre = input("Usuario: ")
if nombre not in users['users']:
print("Usuario no existe.")
return None
pwd = input("Contraseña: ")
info = users['users'][nombre]
if hash_password(pwd, info['salt']) == info['pw
