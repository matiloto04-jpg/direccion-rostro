import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from google.colab.output import eval_js
from IPython.display import display, Javascript
from base64 import b64decode
import PIL
import io

# Función para capturar imagen desde la cámara
def capture_image():
    display(Javascript('''
    async function captureImage() {
        const video = document.createElement('video');
        const canvas = document.createElement('canvas');
        document.body.appendChild(video);
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();

        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        stream.getTracks()[0].stop();
        video.remove();
        return canvas.toDataURL('image/png');
    }
    '''))
    data = eval_js('captureImage()')
    binary = b64decode(data.split(',')[1])
    image = np.array(PIL.Image.open(io.BytesIO(binary)))
    return image

# Mostrar imagen capturada
image = capture_image()
cv2_imshow(image)


+++++++++++++++++++++++++++++++++

# Cargar el clasificador Haar Cascade para la detección de rostro y las partes del rostro
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

# Convertir imagen a escala de grises para facilitar la detección
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detección de rostros
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Dibujar rectángulos alrededor del rostro y las partes del rostro
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

    # Detectar ojos
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Detectar nariz
    nose_cascade = cv2.CascadeClassifier('/content/haarcascade_mcs_nose.xml')

    nose = nose_cascade.detectMultiScale(roi_gray)
    for (nx, ny, nw, nh) in nose:
        cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)

    # Detectar boca
    mouth_cascade = cv2.CascadeClassifier('/content/haarcascade_mcs_mouth.xml')
    mouth = mouth_cascade.detectMultiScale(roi_gray)
    for (mx, my, mw, mh) in mouth:
        cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 255, 0), 2)

# Mostrar imagen con las detecciones
cv2_imshow(image)
