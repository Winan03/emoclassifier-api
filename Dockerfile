# Usa una imagen base de Python 3.10 (NO 3.13)
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . .

# Instala las dependencias del sistema requeridas por librosa y soundfile
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 gcc && rm -rf /var/lib/apt/lists/*

# Instala las dependencias del proyecto
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Descarga el modelo (si usas el script)
RUN python download_model.py

# Expone el puerto que usará tu app (opcional, para Flask por defecto es 5000)
EXPOSE 5000

# Comando para iniciar la app con gunicorn (ajusta si tu archivo principal se llama diferente)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
