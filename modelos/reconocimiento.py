import os
k = cv2.waitKey(1) & 0xFF
if k == ord('s'):
if len(faces) == 0:
print("No se detectó cara.")
continue
x, y, w, h = faces[0]
face_img = gray[y:y + h, x:x + w]
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




def train_recognizer(save_path=RECOGNIZER_FILE, users_data=None):
cascade = cv2.CascadeClassifier(HAAR_CASCADE)
images = []
labels = []
if users_data is None:
from .almacenamiento import load_users
users_data = load_users()


for user, info in users_data["users"].items():
label = info["label"]
user_path = os.path.join(USERS_DIR, user)
if not os.path.isdir(user_path):
continue
for fn in os.listdir(user_path):
if fn.lower().endswith((".png", ".jpg", ".jpeg")):
img = cv2.imread(os.path.join(user_path, fn), cv2.IMREAD_GRAYSCALE)
if img is None:
continue
img_resized = cv2.resize(img, (200, 200))
images.append(img_resized)
labels.append(label)
if len(images) == 0:
print("No hay imágenes para entrenar.")
return
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
recognizer.write(save_path)
print(f"Entrenamiento completo ({len(images)} imágenes).")




def load_recognizer(path=RECOGNIZER_FILE):
if not os.path.exists(path):
return None
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path)
return recognizer
