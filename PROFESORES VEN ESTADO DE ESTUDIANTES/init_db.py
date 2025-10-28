import sqlite3

def init_db():
    conn = sqlite3.connect("votaciones.db")
    cursor = conn.cursor()

    # Crear tabla estudiantes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS estudiantes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        curso TEXT NOT NULL
    )
    """)

    # Crear tabla votos
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS votos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        estudiante_id INTEGER NOT NULL,
        profesor_id INTEGER NOT NULL,
        voto TEXT NOT NULL,
        FOREIGN KEY (estudiante_id) REFERENCES estudiantes (id)
    )
    """)

    # Insertar algunos datos de ejemplo
    cursor.execute("INSERT OR IGNORE INTO estudiantes (id, nombre, curso) VALUES (1, 'Juan Pérez', '10A')")
    cursor.execute("INSERT OR IGNORE INTO estudiantes (id, nombre, curso) VALUES (2, 'María García', '10A')")
    cursor.execute("INSERT OR IGNORE INTO votos (estudiante_id, profesor_id, voto) VALUES (1, 1, 'positivo')")
    cursor.execute("INSERT OR IGNORE INTO votos (estudiante_id, profesor_id, voto) VALUES (2, 1, 'negativo')")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Base de datos inicializada correctamente")