# gunicorn.conf.py - Configuración optimizada para poca RAM

import multiprocessing
import os

# CONFIGURACIÓN DE WORKERS OPTIMIZADA PARA POCA RAM
workers = 1  # Solo 1 worker para minimizar uso de memoria
worker_class = "sync"
worker_connections = 1000

# TIMEOUTS AUMENTADOS para procesos lentos
timeout = 300  # 5 minutos (aumentado de 120s por defecto)
keepalive = 5
graceful_timeout = 300

# CONFIGURACIÓN DE MEMORIA
max_requests = 50  # Reiniciar worker cada 50 requests para limpiar memoria
max_requests_jitter = 10  # Añadir variabilidad
preload_app = True  # Cargar app antes de fork para compartir memoria

# CONFIGURACIÓN DE RED
bind = "0.0.0.0:10000"
backlog = 512

# LOGGING SIMPLIFICADO
loglevel = "warning"  # Solo advertencias y errores
accesslog = "-"  # stdout
errorlog = "-"   # stderr
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" %(D)s'

# CONFIGURACIÓN DE PROCESOS
user = None
group = None
tmp_upload_dir = "/tmp"

# LIMITES DE MEMORIA (si está disponible)
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# CONFIGURACIÓN ESPECÍFICA PARA RENDER/HEROKU
if os.environ.get('PORT'):
    bind = f"0.0.0.0:{os.environ.get('PORT')}"

# HOOK PARA LIMPIAR MEMORIA
def on_exit(server):
    """Limpiar recursos al salir."""
    import gc
    gc.collect()

def worker_exit(server, worker):
    """Limpiar memoria cuando worker termina."""
    import gc
    gc.collect()

def pre_fork(server, worker):
    """Configurar worker antes de fork."""
    import gc
    gc.collect()

# CONFIGURACIÓN ADICIONAL PARA ENTORNOS CON POCA RAM
if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER'):
    # Configuración específica para Railway/Render
    workers = 1
    threads = 1  # Sin threads para simplificar
    worker_tmp_dir = "/dev/shm"  # Usar memoria compartida si está disponible