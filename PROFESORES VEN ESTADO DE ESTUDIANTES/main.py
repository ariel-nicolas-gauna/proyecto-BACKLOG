from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import sqlite3

app = FastAPI()

class Estudiante(BaseModel):
    id: int
    nombre: str
    curso: str
    voto: str

def get_db_connection():
    conn = sqlite3.connect("votaciones.db")
    conn.row_factory = sqlite3.Row
    return conn

def get_current_profesor_id():
    # TODO: Implementar lógica de autenticación real
    return 1

@app.get("/profesor/votantes", response_model=list[Estudiante])
def listar_votantes(profesor_id: int = Depends(get_current_profesor_id)):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Consultamos los votos relacionados con ese profesor
        cursor.execute("""
            SELECT e.id, e.nombre, e.curso, v.voto
            FROM votos v
            JOIN estudiantes e ON e.id = v.estudiante_id
            WHERE v.profesor_id = ?
        """, (profesor_id,))

        resultados = cursor.fetchall()

        if not resultados:
            raise HTTPException(status_code=404, detail="No se encontraron votantes para este profesor")

        return [Estudiante(**dict(row)) for row in resultados]
    finally:
        conn.close()