import os
import gdown

# Coloca aquí los IDs reales de tus archivos (reemplaza los valores de ejemplo)
MODEL_ID = '1gJkGhIY_pZ-tzbpHYCYFn_s4-7v8-vVH'  # <-- ID real de emotion_vgg16.keras
LABEL_ID = '17y1ZoM82NrJkz4nsG1WLkrZk9NoAxJ5Y'  # <-- ID real de label_encoder.pkl

MODEL_PATH = 'models/emotion_vgg16.keras'
LABEL_PATH = 'models/label_encoder.pkl'

os.makedirs('models', exist_ok=True)

# Descargar el modelo (.keras)
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde Google Drive...")
    url = f'https://drive.google.com/uc?id={MODEL_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)
    print("¡Modelo descargado!")
    print("Tamaño del archivo descargado:", os.path.getsize('models/emotion_vgg16.keras'), "bytes")
else:
    print("Modelo ya existe. No se descarga.")

# Descargar el label_encoder (.pkl)
if not os.path.exists(LABEL_PATH):
    print("Descargando label_encoder desde Google Drive...")
    url = f'https://drive.google.com/uc?id={LABEL_ID}'
    gdown.download(url, LABEL_PATH, quiet=False)
    print("¡label_encoder descargado!")
else:
    print("label_encoder ya existe. No se descarga.")
