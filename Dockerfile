
# Usa una imagen base de Python 3.10 (NO 3.13)
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo los archivos necesarios (sin la carpeta models/)
COPY requirements.txt .
COPY app.py .
COPY utils/ ./utils/
COPY templates/ ./templates/
COPY static/ ./static/

# Instala las dependencias del sistema requeridas por librosa y soundfile
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 gcc && rm -rf /var/lib/apt/lists/*

# Instala las dependencias del proyecto
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto que usará tu app
EXPOSE 5000

# Comando para iniciar la app
CMD ["python", "app.py"]