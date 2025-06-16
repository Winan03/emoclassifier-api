import os
import io
import tempfile
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

# --- IMPORTS DE UTILS ---
from utils.audio_to_image import audio_to_mel_spectrogram, validate_audio_file
from utils.preprocess import preprocess_for_model

# Desactivar mensajes de log de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- RUTAS DEL MODELO ---
MODEL_PATH = 'models/emotion_vgg16.keras'
LABELS_PATH = 'models/label_encoder.pkl'

model = None
emotion_labels = None

def load_model_and_labels():
    global model, emotion_labels
    try:
        print(f"Cargando modelo desde: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo cargado exitosamente.")
        print(f"Forma de entrada esperada: {model.input_shape}")

        print(f"Cargando etiquetas desde: {LABELS_PATH}")
        with open(LABELS_PATH, 'rb') as f:
            label_encoder = joblib.load(f)
        emotion_labels = label_encoder.classes_ if hasattr(label_encoder, 'classes_') else label_encoder
        print("Etiquetas cargadas:", emotion_labels)
    except Exception as e:
        print(f"Error al cargar el modelo o las etiquetas: {e}")
        model, emotion_labels = None, None

def spectrogram_to_base64(spectrogram_array):
    try:
        img_array = (spectrogram_array * 255).astype(np.uint8) if spectrogram_array.max() <= 1.0 else np.clip(spectrogram_array, 0, 255).astype(np.uint8)
        if len(img_array.shape) == 2:  # Grayscale a RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error al convertir espectrograma a base64: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or emotion_labels is None:
        return jsonify({'error': 'El modelo o las etiquetas no están cargados.'}), 500

    if not request.files and not request.json:
        return jsonify({'error': 'No se encontraron datos de audio.'}), 400

    audio_file = None
    temp_audio_path = None
    
    try:
        # --- Recibir archivo o base64 ---
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No se seleccionó ningún archivo de audio.'}), 400
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
            except Exception as e:
                return jsonify({'error': f'Error al decodificar audio base64: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Formato de audio no reconocido.'}), 400

        # --- Guardar archivo temporal si es necesario ---
        if audio_file and temp_audio_path is None:
            file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                audio_file.save(tmp.name)
                temp_audio_path = tmp.name

        print(f"Procesando archivo: {temp_audio_path}")

        # --- Validar audio antes de espectrograma ---
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            return jsonify({'error': 'El archivo de audio está vacío o no se pudo guardar.'}), 400

        if not validate_audio_file(temp_audio_path, sr=22050, duration=0.1):
            return jsonify({'error': 'El archivo de audio no es válido o está corrupto.'}), 400

        # --- Paso 1: Audio a espectrograma ---
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
            return jsonify({'error': 'Error al generar el espectrograma del audio.'}), 500

        # --- Paso 2: Preprocesar ---
        processed_spectrogram = preprocess_for_model(
            spectrogram_raw,
            target_size=(128, 128),
            normalize_method='minmax'
        )
        if processed_spectrogram is None:
            return jsonify({'error': 'Error en el preprocesamiento del espectrograma.'}), 500

        # --- Paso 3: Predicción ---
        model_input = np.expand_dims(processed_spectrogram, axis=0)
        predictions = model.predict(model_input, verbose=0)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_index])
        predicted_emotion = emotion_labels[predicted_class_index]

        # --- Paso 4: Espectrograma base64 para frontend ---
        spectrogram_b64 = spectrogram_to_base64(processed_spectrogram)

        # --- Paso 5: Probabilidades ---
        all_probabilities_list = [
            {'emotion': emotion_labels[i], 'probability': float(predictions[0][i])}
            for i in range(len(emotion_labels))
        ]
        all_probabilities_list.sort(key=lambda x: x['probability'], reverse=True)

        # --- Respuesta ---
        response_data = {
            'success': True,
            'emotion': predicted_emotion,
            'confidence': confidence,
            'confidence_percentage': f"{confidence:.1%}",
            'all_probabilities': all_probabilities_list,
            'spectrogram_data': spectrogram_b64,
            'model_info': {
                'input_shape': str(model.input_shape),
                'total_classes': len(emotion_labels),
                'preprocessing_method': 'minmax_normalization'
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
        print(f"❌ Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

    finally:
        # --- Limpiar archivo temporal ---
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"🧹 Archivo temporal eliminado: {temp_audio_path}")
            except Exception as e:
                print(f"⚠️  Error al eliminar archivo temporal: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'server': 'running',
        'model_loaded': model is not None,
        'labels_loaded': emotion_labels is not None,
        'available_emotions': list(emotion_labels) if emotion_labels is not None else [],
        'model_input_shape': str(model.input_shape) if model is not None else None
    }
    return jsonify(status)

@app.route('/test-real', methods=['GET'])
def test_real_audio():
    """Endpoint para probar pipeline con audio real de la carpeta."""
    audio_path = 'tess_data/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav'
    if not os.path.exists(audio_path):
        return jsonify({'error': f'Archivo no encontrado: {audio_path}'}), 404

    if not validate_audio_file(audio_path, sr=22050, duration=0.1):
        return jsonify({'error': 'Archivo inválido.'}), 400

    # Generar espectrograma
    spectrogram_raw = audio_to_mel_spectrogram(audio_path, img_size=(128, 128), sr=22050, duration=3.0)
    if spectrogram_raw is None:
        return jsonify({'error': 'Error generando espectrograma'}), 500

    processed_spec = preprocess_for_model(spectrogram_raw, target_size=(128, 128), normalize_method='minmax')
    if processed_spec is None:
        return jsonify({'error': 'Error preprocesando espectrograma'}), 500

    model_input = np.expand_dims(processed_spec, axis=0)
    predictions = model.predict(model_input, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[0][predicted_class])

    # ¡AQUÍ agregamos los valores crudos!
    return jsonify({
        'file': audio_path,
        'emotion': str(emotion_labels[predicted_class]),
        'confidence': confidence,
        'all_probs': [
            {'emotion': str(emotion_labels[i]), 'prob': float(predictions[0][i])}
            for i in range(len(emotion_labels))
        ],
        'raw_predictions': predictions[0].tolist(),  # <- Esto es el vector crudo
        'emotion_labels': [str(e) for e in emotion_labels]
    })

if __name__ == '__main__':
    print("🚀 Iniciando EmoClassifier...")
    load_model_and_labels()
    if model is None or emotion_labels is None:
        print("❌ ERROR: No se pudo cargar el modelo o las etiquetas.")
        print(f"- Modelo: {MODEL_PATH}")
        print(f"- Labels: {LABELS_PATH}")
    else:
        print("✅ Modelo y etiquetas cargados correctamente.")
        print(f"✅ Emociones disponibles: {list(emotion_labels)}")
        print(f"✅ Input shape del modelo: {model.input_shape}")
    app.run(debug=True, port=5000, host='0.0.0.0')
