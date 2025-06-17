import os
import io
import tempfile
import base64
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Configuración avanzada de logging
def setup_logging(app):
    """Configura el sistema de logging con rotación de archivos y formatos personalizados."""
    
    # Directorio para logs
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configuración básica
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    # Formato personalizado
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para archivo con rotación (max 5 archivos de 10MB cada uno)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'emotion_classifier.log'),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Handler para consola (solo en desarrollo)
    if app.debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
    
    logger.addHandler(file_handler)
    
    # Desactivar logs de librerías específicas
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logger

# Configurar matplotlib antes de importar otras librerías
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Importar funciones desde tus módulos utils
try:
    from utils.audio_to_image import audio_to_mel_spectrogram, validate_audio_file
    from utils.preprocess import preprocess_for_model
    print("✅ Módulos utils importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos utils: {e}")

# Desactivar mensajes de log de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__, template_folder='templates', static_folder='static')

# 2. Configuración segura de CORS para producción
if os.environ.get('FLASK_ENV') == 'production':
    CORS(app, resources={
        r"/predict": {
            "origins": ["https://tudominio.com", "https://www.tudominio.com"],
            "methods": ["POST"],
            "allow_headers": ["Content-Type"]
        },
        r"/health": {
            "origins": "*",
            "methods": ["GET"]
        }
    })
else:
    # Configuración más permisiva para desarrollo
    CORS(app)

# Configurar logging
logger = setup_logging(app)

# Configuración del Modelo y Etiquetas
MODEL_PATH = 'models/emotion_vgg16.tflite'
LABELS_PATH = 'models/label_encoder.pkl'

# Variables globales para el modelo
interpreter = None
input_details = None
output_details = None
emotion_labels = None

def log_system_info():
    """Registra información del sistema y entorno."""
    import platform
    import tensorflow as tf
    
    logger.info("\n" + "="*50)
    logger.info("INICIALIZANDO APLICACIÓN - SYSTEM INFO")
    logger.info(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Sistema Operativo: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Flask: {importlib.metadata.version('flask')}")
    logger.info(f"TensorFlow: {tf.__version__}")
    logger.info(f"Directorio de Trabajo: {os.getcwd()}")
    logger.info(f"Variables de Entorno: FLASK_ENV={os.environ.get('FLASK_ENV')}")
    logger.info("="*50 + "\n")

def load_model_and_labels():
    """Carga el modelo TensorFlow Lite y las etiquetas de las clases."""
    global interpreter, input_details, output_details, emotion_labels
    
    try:
        # Verificar que los archivos existen
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Modelo no encontrado en: {MODEL_PATH}")
            return False
            
        if not os.path.exists(LABELS_PATH):
            logger.error(f"❌ Etiquetas no encontradas en: {LABELS_PATH}")
            return False
        
        # Registrar metadatos del modelo
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        model_mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        logger.info(f"📦 Modelo encontrado: {MODEL_PATH}")
        logger.info(f"   - Tamaño: {model_size:.2f} MB")
        logger.info(f"   - Última modificación: {model_mtime}")
        
        # Cargar modelo TFLite
        logger.info(f"🔍 Cargando modelo TFLite desde: {MODEL_PATH}")
        start_time = datetime.now()
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        load_time = (datetime.now() - start_time).total_seconds()
        
        # Obtener detalles de entrada y salida
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"✅ Modelo TFLite cargado exitosamente en {load_time:.2f}s")
        logger.info(f"   - Forma de entrada: {input_details[0]['shape']}")
        logger.info(f"   - Tipo de datos: {input_details[0]['dtype']}")
        logger.info(f"   - Forma de salida: {output_details[0]['shape']}")
        
        # Cargar etiquetas
        logger.info(f"🏷️ Cargando etiquetas desde: {LABELS_PATH}")
        with open(LABELS_PATH, 'rb') as f:
            label_encoder = joblib.load(f)
        
        if hasattr(label_encoder, 'classes_'):
            emotion_labels = label_encoder.classes_
        else:
            emotion_labels = label_encoder
        
        logger.info("✅ Etiquetas cargadas exitosamente:")
        logger.info(f"   - Emociones disponibles: {list(emotion_labels)}")
        logger.info(f"   - Total de clases: {len(emotion_labels)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al cargar el modelo TFLite o las etiquetas: {str(e)}", 
                    exc_info=True)
        interpreter = None
        input_details = None
        output_details = None
        emotion_labels = None
        return False


def predict_with_tflite(input_data):
    """Realiza predicción usando el modelo TensorFlow Lite."""
    global interpreter, input_details, output_details
    
    try:
        # Asegurar que el input tenga el tipo correcto
        input_data = input_data.astype(np.float32)
        
        # Configurar tensor de entrada
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Ejecutar inferencia
        interpreter.invoke()
        
        # Obtener resultado
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return output_data
        
    except Exception as e:
        logger.error(f"Error durante la predicción TFLite: {e}")
        return None

def spectrogram_to_base64(spectrogram_array):
    """Convierte el array del espectrograma a imagen base64 para el frontend."""
    try:
        # Asegurar valores en rango correcto (0-1) si vienen de preprocess
        if spectrogram_array.max() <= 1.001 and spectrogram_array.min() >= -0.001:
            img_array = (spectrogram_array * 255).astype(np.uint8)
        else:
            # Si por alguna razón no está en [0,1], lo clipamos a 0-255 y convertimos
            img_array = np.clip(spectrogram_array, 0, 255).astype(np.uint8)
        
        # Si es grayscale (H,W), convertir a RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Crear imagen PIL
        img = Image.fromarray(img_array)
        
        # Convertir a base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error al convertir espectrograma a base64: {e}")
        return None

# 🔥 LÍNEA CLAVE: Cargar modelo SIEMPRE, no solo en __main__
logger.info("🚀 Inicializando aplicación Emotion Classifier...")
log_system_info()
logger.info("🔄 Cargando modelo y etiquetas...")
model_loaded = load_model_and_labels()

if not model_loaded:
    logger.critical("🛑 CRÍTICO: No se pudo cargar el modelo. La aplicación no funcionará correctamente.")

@app.before_request
def log_request_info():
    """Registra información de cada solicitud entrante."""
    if request.path == '/health':
        return  # No registrar solicitudes de health check
    
    logger.info(f"📥 Solicitud entrante: {request.method} {request.path}")
    logger.debug(f"Headers: {dict(request.headers)}")
    
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type', '')
        if 'multipart/form-data' in content_type:
            logger.debug("Datos multipart (archivo)")
        elif 'application/json' in content_type:
            logger.debug(f"Datos JSON: {request.json}")
        else:
            logger.debug(f"Datos recibidos: {request.data[:200]}...")  # Primeros 200 caracteres

@app.after_request
def log_response_info(response):
    """Registra información de cada respuesta saliente."""
    if request.path == '/health':
        return response  # No registrar respuestas de health check
    
    logger.info(f"📤 Respuesta saliente: {response.status}")
    
    if response.status_code >= 400:
        logger.error(f"Error {response.status_code}: {response.get_data(as_text=True)[:500]}")
    
    return response

@app.route('/')
def index():
    """Ruta principal que renderiza la página HTML."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error al renderizar index.html: {e}")
        return jsonify({'error': 'Error al cargar la página principal'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Ruta para la predicción de emoción a partir de un archivo de audio real."""
    logger.info("🔮 Iniciando predicción...")
    
    if interpreter is None or emotion_labels is None:
        logger.error("Modelo no cargado correctamente")
        return jsonify({
            'error': 'El modelo o las etiquetas no se han cargado correctamente.',
            'details': 'Reinicia el servidor o verifica los archivos del modelo.',
            'model_path': MODEL_PATH,
            'labels_path': LABELS_PATH
        }), 500

    # Verificar que la solicitud tenga contenido
    if not request.files and not request.json:
        return jsonify({'error': 'No se encontraron datos de audio.'}), 400

    audio_file = None
    temp_audio_path = None
    
    try:
        # Manejar diferentes formatos de entrada de audio
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No se seleccionó ningún archivo de audio.'}), 400
            
            # Verificar extensión del archivo
            allowed_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.webm', '.aac', '.flac'}
            file_ext = os.path.splitext(audio_file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                return jsonify({'error': f'Formato no soportado: {file_ext}. Use: {", ".join(allowed_extensions)}'}), 400
        
        elif request.json and 'audioData' in request.json:
            try:
                audio_data = request.json['audioData']
                # Remover el prefijo data:audio/...;base64, si existe
                if ',' in audio_data:
                    audio_bytes = base64.b64decode(audio_data.split(',')[1])
                else:
                    audio_bytes = base64.b64decode(audio_data)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_bytes)
                    temp_audio_path = tmp.name
            except Exception as e:
                return jsonify({'error': f'Error al decodificar audio base64: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Formato de audio no reconocido. Se esperaba un archivo o base64.'}), 400

        # Guardar archivo temporal si es necesario
        if audio_file and temp_audio_path is None:
            file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                audio_file.save(tmp.name)
                temp_audio_path = tmp.name

        logger.info(f"Procesando archivo de audio temporal: {temp_audio_path}")

        # Verificar que el archivo temporal existe y no está vacío
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            return jsonify({'error': 'El archivo de audio está vacío o no se pudo guardar correctamente.'}), 400

        # PASO 1: Convertir audio a espectrograma Mel usando tu función de utils
        logger.info("🎵 Generando espectrograma Mel...")
        spectrogram_raw = audio_to_mel_spectrogram(
            temp_audio_path, 
            img_size=(128, 128),
            sr=22050,
            duration=3.0,
            n_mels=128,
            n_fft=2048,
            fmax=8000
        )

        if spectrogram_raw is None:
            return jsonify({'error': 'Error al generar el espectrograma del audio. Archivo inválido o corrupto.'}), 500

        logger.info(f"✅ Espectrograma generado: {spectrogram_raw.shape}")
        logger.info(f"   Rango: [{spectrogram_raw.min():.4f}, {spectrogram_raw.max():.4f}]")

        # PASO 2: Preprocesar usando tu función de utils
        logger.info("🔧 Preprocesando espectrograma...")
        processed_spectrogram = preprocess_for_model(
            spectrogram_raw, 
            target_size=(128, 128),
            normalize_method='minmax'  # Usar minmax ya que VGG16 espera [0,1]
        )

        if processed_spectrogram is None:
            return jsonify({'error': 'Error en el preprocesamiento del espectrograma.'}), 500

        logger.info(f"✅ Espectrograma preprocesado: {processed_spectrogram.shape}")
        logger.info(f"   Rango final: [{processed_spectrogram.min():.4f}, {processed_spectrogram.max():.4f}]")

        # PASO 3: Preparar para el modelo (añadir batch dimension)
        model_input = np.expand_dims(processed_spectrogram, axis=0)
        logger.info(f"📊 Input para modelo: {model_input.shape}")

        # PASO 4: Predicción usando TensorFlow Lite
        logger.info("🤖 Realizando predicción...")
        predictions = predict_with_tflite(model_input)
        
        if predictions is None:
            return jsonify({'error': 'Error durante la predicción del modelo.'}), 500
            
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_index])

        # Obtener emoción predicha
        predicted_emotion = emotion_labels[predicted_class_index]

        logger.info(f"🎯 Predicción: {predicted_emotion} (confianza: {confidence:.2%})")

        # PASO 5: Convertir espectrograma a base64 para el frontend
        spectrogram_b64 = spectrogram_to_base64(processed_spectrogram)

        # PASO 6: Preparar todas las probabilidades
        all_probabilities_list = []
        for i, emotion in enumerate(emotion_labels):
            all_probabilities_list.append({
                'emotion': emotion,
                'probability': float(predictions[0][i])
            })
        
        # Ordenar por probabilidad descendente
        all_probabilities_list.sort(key=lambda x: x['probability'], reverse=True)

        # RESPUESTA FINAL
        response_data = {
            'success': True,
            'emotion': predicted_emotion,
            'confidence': confidence,
            'confidence_percentage': f"{confidence:.1%}",
            'all_probabilities': all_probabilities_list,
            'spectrogram_data': spectrogram_b64,
            'model_info': {
                'input_shape': str(input_details[0]['shape']),
                'total_classes': len(emotion_labels),
                'preprocessing_method': 'minmax_normalization',
                'model_type': 'TensorFlow Lite'
            },
            'audio_info': {
                'duration_seconds': 3.0,
                'sample_rate_hz': 22050,
                'spectrogram_size': '128x128',
                'processed': True
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Error interno del servidor: {str(e)}',
            'type': type(e).__name__,
            'traceback': traceback.format_exc() if app.debug else None
        }), 500
    
    finally:
        # Limpiar archivo temporal
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"🧹 Archivo temporal eliminado: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"⚠️  Error al eliminar archivo temporal: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servidor y modelo."""
    status = {
        'server': 'running',
        'model_loaded': interpreter is not None,
        'labels_loaded': emotion_labels is not None,
        'available_emotions': list(emotion_labels) if emotion_labels is not None else [],
        'model_input_shape': str(input_details[0]['shape']) if input_details is not None else None,
        'model_type': 'TensorFlow Lite',
        'model_path_exists': os.path.exists(MODEL_PATH),
        'labels_path_exists': os.path.exists(LABELS_PATH),
        'working_directory': os.getcwd(),
        'environment': {
            'MPLCONFIGDIR': os.environ.get('MPLCONFIGDIR'),
            'HOME': os.environ.get('HOME'),
            'USER': os.environ.get('USER')
        }
    }
    return jsonify(status)

@app.route('/test-pipeline', methods=['GET'])
def test_pipeline():
    """Endpoint para probar el pipeline completo con audio de ejemplo."""
    try:
        # Crear un audio de prueba simple
        import soundfile as sf
        import numpy as np
        
        # Generar audio de prueba (tono simple)
        samplerate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(samplerate * duration), False)
        
        # Mezcla de frecuencias
        test_audio = (0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
                     0.2 * np.sin(2 * np.pi * 880 * t) +  # A5
                     0.1 * np.random.normal(0, 0.1, len(t)))  # Ruido suave
        
        # Guardar audio temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            sf.write(tmp.name, test_audio, samplerate)
            test_audio_path = tmp.name
        
        # Probar pipeline completo
        spectrogram_raw = audio_to_mel_spectrogram(test_audio_path)
        if spectrogram_raw is None:
            return jsonify({'error': 'Error generando espectrograma de prueba'}), 500
            
        processed_spec = preprocess_for_model(spectrogram_raw)
        if processed_spec is None:
            return jsonify({'error': 'Error preprocesando espectrograma de prueba'}), 500
            
        # Predicción de prueba usando TFLite
        model_input = np.expand_dims(processed_spec, axis=0)
        predictions = predict_with_tflite(model_input)
        
        if predictions is None:
            return jsonify({'error': 'Error en predicción de prueba'}), 500
            
        predicted_class = np.argmax(predictions)
        
        # Limpiar archivo temporal
        os.remove(test_audio_path)
        
        return jsonify({
            'pipeline_test': 'success',
            'spectrogram_shape': spectrogram_raw.shape,
            'processed_shape': processed_spec.shape,
            'model_input_shape': model_input.shape,
            'predicted_emotion': emotion_labels[predicted_class],
            'confidence': float(predictions[0][predicted_class]),
            'model_type': 'TensorFlow Lite'
        })
        
    except Exception as e:
        return jsonify({'pipeline_test': 'failed', 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {error}")
    return jsonify({'error': 'Error interno del servidor'}), 500

# 🎯 PUNTO CLAVE: Manejar inicialización para desarrollo local
if __name__ == '__main__':
    logger.info("🚀 Iniciando EmoClassifier en modo desarrollo...")
        
    if interpreter is None or emotion_labels is None:
        logger.error("❌ ERROR: No se pudo cargar el modelo o las etiquetas.")
        logger.error("Verifica que los archivos existan en las rutas especificadas:")
        logger.error(f"- Modelo: {MODEL_PATH}")
        logger.error(f"- Labels: {LABELS_PATH}")
    else:
        logger.info("✅ Modelo y etiquetas cargados correctamente.")
        logger.info(f"✅ Emociones disponibles: {list(emotion_labels)}")
        logger.info(f"✅ Input shape del modelo: {input_details[0]['shape']}")
    
    # Iniciar Flask en modo desarrollo
    app.run(debug=True, port=5000, host='0.0.0.0')