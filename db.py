import sqlite3
import numpy as np
from scipy.spatial.distance import cosine

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            image_path TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

# Сохранение эмбеддинга
def save_embedding(image_path, embedding):
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO faces (image_path, embedding) VALUES (?, ?)',
                   (image_path, embedding.tobytes()))
    conn.commit()
    conn.close()

# Поиск похожих лиц
def get_top_similar_faces(embedding, top_n=3):
    conn = sqlite3.connect('database/faces.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_path, embedding FROM faces")
    rows = cursor.fetchall()
    
    similarities = []
    for row in rows:
        db_embedding = np.frombuffer(row[1], dtype=np.float32)
        similarity = 1 - cosine(embedding, db_embedding)
        similarities.append((similarity, row[0]))
    
    similarities.sort(reverse=True)
    conn.close()
    
    return similarities[:top_n]

if __name__ == '__main__':
    init_db()