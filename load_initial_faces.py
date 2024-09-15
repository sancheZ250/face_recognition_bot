import os
import numpy as np
from PIL import Image
from face_recognition import extract_embedding
from db import save_embedding

# Путь к папке с изображениями лиц
FACE_IMAGES_DIR = 'face_images'

def load_faces():
    # Проверка, существует ли папка с изображениями
    if not os.path.exists(FACE_IMAGES_DIR):
        print(f'Папка {FACE_IMAGES_DIR} не существует.')
        return

    # Проход по всем изображениям в папке
    for filename in os.listdir(FACE_IMAGES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(FACE_IMAGES_DIR, filename)
            
            # Открытие изображения
            try:
                face_image = Image.open(image_path)
                embedding = extract_embedding(face_image)
                
                # Сохранение эмбеддинга в базу данных
                save_embedding(image_path, embedding)
                print(f'Эмбеддинг для {filename} сохранен в базе данных.')
            
            except Exception as e:
                print(f'Ошибка обработки {filename}: {e}')

if __name__ == '__main__':
    load_faces()
