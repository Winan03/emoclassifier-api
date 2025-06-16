print("===> INICIANDO APP.PY <===")
import os
import io
import tempfile
import base64
import requests
import gc  # Garbage collector
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

# --- IMPORTS DE UTILS ---
from utils.audio_to_image import audio_to_mel_spectrogram, validate_audio_file
from utils.preprocess import preprocess_for_model

# Configuración optimizada de TensorFlow para memoria
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.optimizer.set_memory_growth_enabled = True

# Configurar límite de memoria para TensorFlow
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
except RuntimeError:
    pass

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- URLS DE S3 ---
MODEL_S3_URL = 'https://emoclassifier.s3.us-east-2.amazonaws.com/emotion_vgg16.keras'
LABELS_S3_URL = 'https://emoclassifier.s3.us-east-2.amazonaws.com/label_encoder.pkl'

model = None
emotion_labels = None

def load_model_and_labels():
    global model, emotion_labels
    
    print("🧠 Liberando memoria antes de cargar modelo...")
    gc.collect()
    
    try:
        # --- CARGAR ETIQUETAS PRIMERO (más liviano) ---
        print(f"📥 Descargando etiquetas desde S3: {LABELS_S3_URL}")
        labels_response = requests.get(LABELS_S3_URL, timeout=30, stream=True)
        labels_response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_labels:
            for chunk in labels_response.iter_content(chunk_size=8192):
                tmp_labels.write(chunk)
            tmp_labels_path = tmp_labels.name
        
        with open(tmp_labels_path, 'rb') as f:
            label_encoder = joblib.load(f)
        emotion_labels = label_encoder.classes_ if hasattr(label_encoder, 'classes_') else label_encoder
        print("✅ Etiquetas cargadas:", emotion_labels)
        os.remove(tmp_labels_path)
        
        # Limpiar memoria después de etiquetas
        del labels_response
        gc.collect()
        
        # --- CARGAR MODELO CON STREAMING ---
        print(f"📥 Descargando modelo desde S3: {MODEL_S3_URL}")
        model_response = requests.get(MODEL_S3_URL, timeout=180, stream=True)
        model_response.raise_for_status()
        
        # Usar streaming para evitar cargar todo en memoria
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_model:
            for chunk in model_response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_model.write(chunk)
            tmp_model_path = tmp_model.name
        
        # Limpiar respuesta HTTP inmediatamente
        del model_response
        gc.collect()
        
        # Cargar modelo con configuración de memoria optimizada
        print("🤖 Cargando modelo en TensorFlow...")
        model = tf.keras.models.load_model(tmp_model_path, compile=False)  # No compilar para ahorrar memoria
        print("✅ Modelo cargado exitosamente desde S3.")
        print(f"📊 Input shape: {model.input_shape}")
        
        # Limpiar archivo temporal inmediatamente
        os.remove(tmp_model_path)
        gc.collect()
        
        print("🎯 Optimizando modelo para inferencia...")
        # Hacer una predicción dummy para optimizar el modelo
        dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        del dummy_input
        gc.collect()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de red al descargar desde S3: {e}")
        model, emotion_labels = None, None
    except Exception as e:
        print(f"❌ Error al cargar modelo/etiquetas: {e}")
        model, emotion_labels = None, None
    finally:
        # Limpieza final
        gc.collect()

# ⭐ Carga lazy del modelo (solo cuando sea necesario)
def ensure_model_loaded():
    global model, emotion_labels
    if model is None or emotion_labels is None:
        print("🔄 Cargando modelo bajo demanda...")
        load_model_and_labels()
    return model is not None and emotion_labels is not None

def spectrogram_to_base64(spectrogram_array):
    try:
        img_array = (spectrogram_array * 255).astype(np.uint8) if spectrogram_array.max() <= 1.0 else np.clip(spectrogram_array, 0, 255).astype(np.uint8)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=True, quality=85)  # Reducir calidad
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error al convertir espectrograma: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Cargar modelo solo cuando se necesite
    if not ensure_model_loaded():
        return jsonify({'error': 'El modelo no pudo ser cargado.'}), 500

    if not request.files and not request.json:
        return jsonify({'error': 'No se encontraron datos de audio.'}), 400

    audio_file = None
    temp_audio_path = None
    
    try:
        # --- Procesar entrada de audio ---
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No se seleccionó archivo.'}), 400
            allowed_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.webm', '.aac', '.flac'}
            file_ext = os.path.splitext(audio_file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                return jsonify({'error': f'Formato no soportado: {file_ext}'}), 400
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
                del audio_bytes
            except Exception as e:
                return jsonify({'error': f'Error decodificando audio: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Formato de audio no reconocido.'}), 400

        # --- Guardar archivo temporal ---
        if audio_file and temp_audio_path is None:
            file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                audio_file.save(tmp.name)
                temp_audio_path = tmp.name

        # --- Validaciones ---
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            return jsonify({'error': 'Archivo de audio vacío.'}), 400

        if not validate_audio_file(temp_audio_path, sr=22050, duration=0.1):
            return jsonify({'error': 'Archivo de audio inválido.'}), 400

        # --- Pipeline de procesamiento ---
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
            return jsonify({'error': 'Error generando espectrograma.'}), 500

        processed_spectrogram = preprocess_for_model(
            spectrogram_raw,
            target_size=(128, 128),
            normalize_method='minmax'
        )
        if processed_spectrogram is None:
            return jsonify({'error': 'Error en preprocesamiento.'}), 500

        # --- Predicción ---
        model_input = np.expand_dims(processed_spectrogram, axis=0).astype(np.float32)
        predictions = model.predict(model_input, verbose=0)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_index])
        predicted_emotion = emotion_labels[predicted_class_index]

        # --- Limpiar memoria ---
        del model_input, spectrogram_raw
        gc.collect()

        # --- Respuesta optimizada ---
        response_data = {
            'success': True,
            'emotion': predicted_emotion,
            'confidence': confidence,
            'confidence_percentage': f"{confidence:.1%}",
            'all_probabilities': [
                {'emotion': emotion_labels[i], 'probability': float(predictions[0][i])}
                for i in range(len(emotion_labels))
            ][:5],  # Solo top 5 para ahorrar memoria
            'spectrogram_data': spectrogram_to_base64(processed_spectrogram),
            'model_info': {
                'input_shape': str(model.input_shape),
                'total_classes': len(emotion_labels)
            }
        }
        
        del processed_spectrogram, predictions
        gc.collect()
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"❌ Error durante predicción: {e}")
        gc.collect()
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

    finally:
        # Limpieza de archivos temporales
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as e:
                print(f"⚠️ Error eliminando archivo temporal: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'server': 'running',
        'model_loaded': model is not None,
        'labels_loaded': emotion_labels is not None,
        'model_source': 'S3_LAZY_LOADING'
    }
    if emotion_labels is not None:
        status['available_emotions'] = list(emotion_labels)
    return jsonify(status)

# NO cargar modelo al inicio para ahorrar memoria
print("🚀 EmoClassifier iniciado con carga lazy del modelo")
print("💡 El modelo se cargará en la primera predicción")

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')