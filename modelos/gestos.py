import math




def distancia(a, b):
return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)




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
