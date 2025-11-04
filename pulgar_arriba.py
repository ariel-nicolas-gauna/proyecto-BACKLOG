import cv2
import mediapipe as mp
import math
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def distancia(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def detectar_pulgar_arriba(landmarks):
    pulgar_tip = landmarks[4]
    pulgar_ip = landmarks[3]
    pulgar_mcp = landmarks[2]
    indice_mcp = landmarks[5]
    meñique_mcp = landmarks[17]

    dx = pulgar_tip.x - pulgar_mcp.x
    dy = pulgar_tip.y - pulgar_mcp.y
    angulo = math.degrees(math.atan2(dy, dx))

    apunta_arriba = -160 < angulo < -20

    dedos_tips = [8, 12, 16, 20]
    doblados = all(
        distancia(landmarks[i], landmarks[0]) < distancia(landmarks[i - 2], landmarks[0])
        for i in dedos_tips
    )

    mano_recta = indice_mcp.x < pulgar_mcp.x < meñique_mcp.x or meñique_mcp.x < pulgar_mcp.x < indice_mcp.x

    return apunta_arriba and doblados and mano_recta


nombre = input("Ingrese su nombre: ")
rol = input("Ingrese su rol (por ejemplo 'estudiante', 'docente', etc.): ").strip().lower()

print("\nIniciando cámara... haga el gesto 👍 para confirmar si es estudiante.\n")

cam = cv2.VideoCapture(0)
confirmado = False
ultimo_gesto = time.time()

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            print("No se puede acceder a la cámara. Verificá DroidCam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if detectar_pulgar_arriba(hand_landmarks.landmark):
                    if rol == "estudiante":
                        cv2.putText(frame, f"Gesto correcto 👍", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        cv2.putText(frame, f"{nombre} confirmado como estudiante", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        confirmado = True
                        ultimo_gesto = time.time()
                    else:
                        cv2.putText(frame, f"{nombre} NO es estudiante", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                else:
                    if time.time() - ultimo_gesto > 2:
                        cv2.putText(frame, "Haga el gesto de pulgar arriba 👍", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if confirmado and time.time() - ultimo_gesto < 2:
            cv2.rectangle(frame, (20, 20), (620, 120), (0, 255, 0), 3)

        cv2.imshow("Reconocimiento de gesto - DroidCam", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cam.release()
cv2.destroyAllWindows()