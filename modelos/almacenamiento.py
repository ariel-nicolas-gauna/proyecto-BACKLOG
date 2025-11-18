import json
import os


USERS_FILE = "users.json"
USERS_DIR = "users"


os.makedirs(USERS_DIR, exist_ok=True)
if not os.path.exists(USERS_FILE):
with open(USERS_FILE, "w") as f:
json.dump({"next_label": 0, "users": {}}, f)




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
