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
import importlib.metadata
import gc  # Para forzar garbage collection

# CONFIGURACIÓN DE MEMORIA OPTIMIZADA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Solo errores críticos
tf.get_logger().setLevel('ERROR')

# Limitar uso de memoria de TensorFlow
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# 1. Configuración de logging SIMPLIFICADA (menos memoria)
def setup_logging(app):
    """Configuración ligera de logging."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.WARNING,  # Solo advertencias y errores
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logger = logging.getLogger()
    
    # Handler más pequeño
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=2*1024*1024,  # Solo 2MB por archivo
        backupCount=2,  # Solo 2 backups
        encoding='utf-8'
    )
    file_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    
    # Silenciar librerías
    for lib in ['matplotlib', 'PIL', 'tensorflow', 'urllib3']:
        logging.getLogger(lib).setLevel(logging.ERROR)
    
    return logger

# Configurar matplotlib con configuración mínima
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Importar funciones utils
try:
    from utils.audio_to_image import audio_to_mel_spectrogram, validate_audio_file
    from utils.preprocess import preprocess_for_model
    print("✅ Módulos utils importados")
except ImportError as e:
    print(f"❌ Error importando utils: {e}")

app = Flask(__name__, template_folder='templates', static_folder='static')

# CORS simplificado
if os.environ.get('FLASK_ENV') == 'production':
    CORS(app, resources={r"/predict": {"origins": "*", "methods": ["POST"]}})
else:
    CORS(app)

logger = setup_logging(app)

# Configuración del modelo
MODEL_PATH = 'models/emotion_vgg16.tflite'
LABELS_PATH = 'models/label_encoder.pkl'

# Variables globales
interpreter = None
input_details = None
output_details = None
emotion_labels = None

def load_model_and_labels():
    """Carga optimizada del modelo."""
    global interpreter, input_details, output_details, emotion_labels
    
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            logger.error("Archivos del modelo no encontrados")
            return False
        
        # Cargar modelo con configuración de memoria limitada
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH,
            num_threads=1  # Limitar threads para reducir memoria
        )
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Cargar etiquetas
        with open(LABELS_PATH, 'rb') as f:
            label_encoder = joblib.load(f)
        
        emotion_labels = getattr(label_encoder, 'classes_', label_encoder)
        
        logger.warning(f"✅ Modelo cargado: {len(emotion_labels)} clases")
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return False

def predict_with_tflite(input_data):
    """Predicción optimizada."""
    try:
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return None

def spectrogram_to_base64_optimized(spectrogram_array):
    """Conversión optimizada a base64."""
    try:
        # Reducir calidad para ahorrar memoria
        if spectrogram_array.max() <= 1.001:
            img_array = (spectrogram_array * 255).astype(np.uint8)
        else:
            img_array = np.clip(spectrogram_array, 0, 255).astype(np.uint8)
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Crear imagen más pequeña
        img = Image.fromarray(img_array)
        # Reducir tamaño si es muy grande
        if img.size[0] > 256 or img.size[1] > 256:
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70, optimize=True) 
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Limpiar memoria
        del img, buffer, img_array
        gc.collect()
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error en base64: {e}")
        return None

# Cargar modelo al inicio
logger.warning("🚀 Iniciando aplicación optimizada...")
model_loaded = load_model_and_labels()

if not model_loaded:
    logger.error("🛑 Modelo no cargado")

@app.before_request
def log_request_info():
    """Log mínimo de requests."""
    if request.path not in ['/health', '/favicon.ico']:
        logger.warning(f"📥 {request.method} {request.path}")

@app.route('/')
def index():
    """Página principal."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error renderizando: {e}")
        return jsonify({'error': 'Error cargando página'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predicción OPTIMIZADA para memoria."""
    logger.warning("🔮 Iniciando predicción optimizada...")
    
    if interpreter is None or emotion_labels is None:
        return jsonify({'error': 'Modelo no cargado'}), 500

    if not request.files and not request.json:
        return jsonify({'error': 'No hay datos de audio'}), 400

    temp_audio_path = None
    
    try:
        # OPTIMIZACIÓN 1: Procesar audio directamente sin validaciones extra
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'Archivo vacío'}), 400
            
            # Guardar con extensión fija para simplificar
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                audio_file.save(tmp.name)
                temp_audio_path = tmp.name
        
        elif request.json and 'audioData' in request.json:
            try:
                audio_data = request.json['audioData']
                if ',' in audio_data:
                    audio_bytes = base64.b64decode(audio_data.split(',')[1])
                else:
                    audio_bytes = base64.b64decode(audio_data)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_bytes)
                    temp_audio_path = tmp.name
                    
                # Limpiar inmediatamente
                del audio_bytes
                gc.collect()
                
            except Exception as e:
                return jsonify({'error': f'Error decodificando: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Formato no reconocido'}), 400

        # OPTIMIZACIÓN 2: Parámetros reducidos para el espectrograma
        logger.warning("🎵 Generando espectrograma...")
        spectrogram_raw = audio_to_mel_spectrogram(
            temp_audio_path, 
            img_size=(64, 64),  # REDUCIDO de 128x128 a 64x64
            sr=16000,           # REDUCIDO de 22050 a 16000
            duration=2.0,       # REDUCIDO de 3.0 a 2.0 segundos
            n_mels=64,          # REDUCIDO de 128 a 64
            n_fft=1024,         # REDUCIDO de 2048 a 1024
            fmax=6000           # REDUCIDO de 8000 a 6000
        )

        if spectrogram_raw is None:
            return jsonify({'error': 'Error generando espectrograma'}), 500

        logger.warning(f"✅ Espectrograma: {spectrogram_raw.shape}")

        # OPTIMIZACIÓN 3: Redimensionar antes del preprocesamiento
        if spectrogram_raw.shape != (128, 128, 3):
            from PIL import Image as PILImage
            # Convertir a PIL y redimensionar
            if len(spectrogram_raw.shape) == 2:
                spec_img = PILImage.fromarray((spectrogram_raw * 255).astype(np.uint8), mode='L')
                spec_img = spec_img.convert('RGB')
            else:
                spec_img = PILImage.fromarray((spectrogram_raw * 255).astype(np.uint8))
            
            spec_img = spec_img.resize((128, 128), PILImage.Resampling.LANCZOS)
            spectrogram_raw = np.array(spec_img).astype(np.float32) / 255.0
            
            del spec_img
            gc.collect()

        # OPTIMIZACIÓN 4: Preprocesamiento simplificado
        logger.warning("🔧 Preprocesando...")
        processed_spectrogram = preprocess_for_model(
            spectrogram_raw, 
            target_size=(128, 128),
            normalize_method='minmax'
        )

        if processed_spectrogram is None:
            return jsonify({'error': 'Error preprocesando'}), 500

        # Limpiar espectrograma raw inmediatamente
        del spectrogram_raw
        gc.collect()

        # OPTIMIZACIÓN 5: Predicción
        model_input = np.expand_dims(processed_spectrogram, axis=0)
        logger.warning("🤖 Prediciendo...")
        
        predictions = predict_with_tflite(model_input)
        
        if predictions is None:
            return jsonify({'error': 'Error en predicción'}), 500
            
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_index])
        predicted_emotion = emotion_labels[predicted_class_index]

        logger.warning(f"🎯 Resultado: {predicted_emotion} ({confidence:.2%})")

        # OPTIMIZACIÓN 6: Generar base64 de forma eficiente
        spectrogram_b64 = spectrogram_to_base64_optimized(processed_spectrogram)

        # OPTIMIZACIÓN 7: Solo top 5 probabilidades para reducir payload
        top_probabilities = []
        top_indices = np.argsort(predictions[0])[::-1][:5]  # Top 5
        for i in top_indices:
            top_probabilities.append({
                'emotion': emotion_labels[i],
                'probability': float(predictions[0][i])
            })

        # Limpiar memoria antes de responder
        del model_input, processed_spectrogram, predictions
        gc.collect()

        # RESPUESTA OPTIMIZADA
        response_data = {
            'success': True,
            'emotion': predicted_emotion,
            'confidence': confidence,
            'confidence_percentage': f"{confidence:.1%}",
            'top_probabilities': top_probabilities,  # Solo top 5
            'spectrogram_data': spectrogram_b64,
            'model_info': {
                'total_classes': len(emotion_labels),
                'model_type': 'TensorFlow Lite Optimized'
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        # Forzar limpieza en caso de error
        gc.collect()
        return jsonify({
            'error': f'Error: {str(e)}',
            'type': type(e).__name__
        }), 500
    
    finally:
        # CRÍTICO: Limpiar archivo temporal SIEMPRE
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.warning(f"🧹 Temporal eliminado")
            except Exception as e:
                logger.error(f"Error eliminando: {e}")
        
        # Forzar garbage collection
        gc.collect()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check simplificado."""
    return jsonify({
        'server': 'running',
        'model_loaded': interpreter is not None,
        'emotions_count': len(emotion_labels) if emotion_labels else 0,
        'memory_optimized': True
    })

# Error handlers minimalistas
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'No encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    gc.collect()  # Limpiar memoria en errores
    return jsonify({'error': 'Error interno'}), 500

if __name__ == '__main__':
    if interpreter is None or emotion_labels is None:
        logger.error("❌ Modelo no cargado")
    else:
        logger.warning("✅ Aplicación lista")
    
    app.run(debug=False, port=5000, host='0.0.0.0')  