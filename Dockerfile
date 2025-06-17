# Usar Python 3.10 que es compatible con TensorFlow 2.11.0
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
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
    # Para descargas (curl para healthcheck)
    curl \
    # Limpieza
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip y instalar wheel para builds más rápidos
RUN pip install --upgrade pip wheel setuptools

# Copiar requirements.txt primero
COPY requirements.txt .

# Instalar TensorFlow primero (es la dependencia más pesada)
RUN pip install --no-cache-dir tensorflow==2.11.0

# Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios si no existen
RUN mkdir -p models static/css static/js templates utils logs

# Verificar que los archivos críticos existen
RUN echo "Verificando estructura de archivos..." && \
    ls -la . && \
    ls -la models/ || echo "⚠️ Directorio models/ vacío - asegúrate de incluir tus archivos .tflite y .pkl" && \
    ls -la utils/ || echo "⚠️ Directorio utils/ vacío"

# Configurar variables de entorno
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2
# Optimizaciones para TensorFlow
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=""

# Crear usuario no-root para mayor seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Exponer el puerto
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Comando para ejecutar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--max-requests", "100", "--preload", "app:app"]