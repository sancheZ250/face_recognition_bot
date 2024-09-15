import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from db import init_db, save_embedding, get_top_similar_faces
from face_recognition import process_face, extract_embedding
from PIL import Image
import numpy as np

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Инициализация бота
API_TOKEN = os.getenv('API_TOKEN')
REQUEST_TIMEOUT = 60

# Логирование
logging.basicConfig(level=logging.INFO)

async def start(update: Update, context):
    """Отправляет приветственное сообщение при команде /start"""
    await update.message.reply_text("Привет! Отправь мне фото, и я найду похожие лица в базе данных.")

async def handle_photo(update: Update, context):
    try:
        # Получение фотографии из сообщения
        photo = update.message.photo[-1]
        file = await photo.get_file()
        
        # Создание пути для сохранения файла
        photo_path = f'user_images/{photo.file_id}.jpg'
        if not os.path.exists('user_images'):
            os.makedirs('user_images')

        # Скачать файл
        await file.download_to_drive(photo_path)
        
        # Открыть изображение с помощью PIL
        face_image = Image.open(photo_path)

        # Обнаружение лица
        detected_face, error_message = process_face(photo_path)
        if detected_face is None:
            await update.message.reply_text(error_message)
            return
        
        # Обработать изображение и получить embedding
        embedding = extract_embedding(face_image) # заменить detected_face -> face_image, если хотим искать схожесть по всей фотографии, а не по обрезанному квадрату лица, и наоборот.
        
        # Найти топ 3 похожих лиц
        top_similar_faces = get_top_similar_faces(embedding)
        
        # Если похожие лица найдены, отправляем их пользователю
        if not top_similar_faces:
            await update.message.reply_text('Не удалось найти похожие лица.')
        else:
            for i, (similarity, image_path) in enumerate(top_similar_faces):
                # Преобразование похожести в процент
                similarity_percentage = round(similarity * 100, 2)
                
                # Отправляем фото
                with open(image_path, 'rb') as img_file:
                    await update.message.reply_photo(photo=img_file, caption=f'Схожесть: {similarity_percentage}%')
        
    except Exception as e:
        # Отправка сообщения об ошибке
        error_message = f'Произошла ошибка: {str(e)}'
        await update.message.reply_text(error_message)




def main():
    """Запуск бота"""
    # Инициализация базы данных
    init_db()

    # Создаем объект приложения бота
    application = Application.builder().token(API_TOKEN).read_timeout(REQUEST_TIMEOUT).write_timeout(REQUEST_TIMEOUT).build()

    # Команды и обработчики сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main()
