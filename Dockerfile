# Usar imagen base oficial de Python 3.11 slim
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Crear usuario no-root ANTES de instalar dependencias
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno para evitar errores de permisos
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV FONTCONFIG_PATH=/tmp/fontconfig
ENV HOME=/tmp

# Crear directorios temporales con permisos correctos
RUN mkdir -p /tmp/matplotlib /tmp/fontconfig && \
    chmod 755 /tmp/matplotlib /tmp/fontconfig

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p models static/css static/js templates utils logs

# Cambiar propietario de todos los archivos a appuser
RUN chown -R appuser:appuser /app && \
    chown -R appuser:appuser /tmp/matplotlib && \
    chown -R appuser:appuser /tmp/fontconfig

# Verificar estructura de archivos
RUN echo "Verificando estructura de archivos..." && \
    ls -la . && \
    ls -la models/ && \
    ls -la utils/ && \
    echo "✅ Estructura verificada"

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 10000

# Comando de inicio con gunicorn optimizado
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "--preload", "app:app"]