from models.reconocimiento import capture_faces_for_user, train_recognizer, load_recognizer
from models.almacenamiento import load_users




def train_command():
data = load_users()
train_recognizer(users_data=data)




def capture_command(nombre, cam_index=0):
capture_faces_for_user(nombre, cam_index=cam_index)




def get_recognizer(path=None):
return load_recognizer(path) if path else load_recognizer()
