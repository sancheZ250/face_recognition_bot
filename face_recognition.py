import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
# Модель для извлечения признаков (FaceNet можно заменить на другую модель)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Преобразование изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def process_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, 'Лицо не найдено.'
    elif len(faces) > 1:
        return None, 'Обнаружено несколько лиц.'
    
    (x, y, w, h) = faces[0]
    face_image = image[y:y+h, x:x+w]
    
    # Преобразование в формат PIL Image
    face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    
    return face_image_pil, None


# Извлечение эмбеддинга
def extract_embedding(face_image):
    face_tensor = preprocess(face_image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_tensor).numpy().flatten()
    return embedding
