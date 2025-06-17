# Usar una imagen base de Python con soporte para bibliotecas científicas
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para audio y multimedia
RUN apt-get update && apt-get install -y \
    # Para librosa y procesamiento de audio
    libsndfile1 \
    ffmpeg \
    # Para OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Para compilación de algunas dependencias
    gcc \
    g++ \
    # Limpieza
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip install --upgrade pip

# Copiar requirements.txt primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios si no existen
RUN mkdir -p models static/css static/js templates utils

# Configurar variables de entorno
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

# Exponer el puerto
EXPOSE 5000

# Verificar que los archivos críticos existen (opcional, para debugging)
RUN ls -la models/ || echo "Directorio models no encontrado"
RUN ls -la utils/ || echo "Directorio utils no encontrado"

# Comando para ejecutar la aplicación
# Usar gunicorn para producción
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--max-requests", "100", "app:app"]

# Alternativa para desarrollo (comentar la línea anterior y descomentar esta)
# CMD ["python", "app.py"]