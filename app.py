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

# Importar funciones desde tus módulos utils
from utils.audio_to_image import audio_to_mel_spectrogram, validate_audio_file
from utils.preprocess import preprocess_for_model

# Desactivar mensajes de log de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuración del Modelo y Etiquetas
MODEL_PATH = 'models/emotion_vgg16.tflite'
LABELS_PATH = 'models/label_encoder.pkl'

# Variables globales para el modelo
interpreter = None
input_details = None
output_details = None
emotion_labels = None

def load_model_and_labels():
    """Carga el modelo TensorFlow Lite y las etiquetas de las clases."""
    global interpreter, input_details, output_details, emotion_labels
    
    try:
        # Cargar modelo TFLite
        print(f"Cargando modelo TFLite desde: {MODEL_PATH}")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Obtener detalles de entrada y salida
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Modelo TFLite cargado exitosamente.")
        print(f"Forma de entrada esperada: {input_details[0]['shape']}")
        print(f"Tipo de datos de entrada: {input_details[0]['dtype']}")
        print(f"Forma de salida esperada: {output_details[0]['shape']}")
        
        # Cargar etiquetas
        print(f"Cargando etiquetas desde: {LABELS_PATH}")
        with open(LABELS_PATH, 'rb') as f:
            label_encoder = joblib.load(f)
        
        if hasattr(label_encoder, 'classes_'):
            emotion_labels = label_encoder.classes_
        else:
            emotion_labels = label_encoder
        
        print("Etiquetas cargadas exitosamente:")
        print(emotion_labels)
        
    except Exception as e:
        print(f"Error al cargar el modelo TFLite o las etiquetas: {e}")
        interpreter = None
        input_details = None
        output_details = None
        emotion_labels = None

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
        print(f"Error durante la predicción TFLite: {e}")
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
        print(f"Error al convertir espectrograma a base64: {e}")
        return None

print("🚀 Cargando modelo y etiquetas al iniciar servidor...")
load_model_and_labels()

@app.route('/')
def index():
    """Ruta principal que renderiza la página HTML."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Ruta para la predicción de emoción a partir de un archivo de audio real."""
    
    if interpreter is None or emotion_labels is None:
        return jsonify({'error': 'El modelo o las etiquetas no se han cargado correctamente. Reinicia el servidor.'}), 500

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

        print(f"Procesando archivo de audio temporal: {temp_audio_path}")

        # Verificar que el archivo temporal existe y no está vacío
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            return jsonify({'error': 'El archivo de audio está vacío o no se pudo guardar correctamente.'}), 400

        # PASO 1: Convertir audio a espectrograma Mel usando tu función de utils
        print("🎵 Generando espectrograma Mel...")
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

        print(f"✅ Espectrograma generado: {spectrogram_raw.shape}")
        print(f"   Rango: [{spectrogram_raw.min():.4f}, {spectrogram_raw.max():.4f}]")

        # PASO 2: Preprocesar usando tu función de utils
        print("🔧 Preprocesando espectrograma...")
        processed_spectrogram = preprocess_for_model(
            spectrogram_raw, 
            target_size=(128, 128),
            normalize_method='minmax'  # Usar minmax ya que VGG16 espera [0,1]
        )

        if processed_spectrogram is None:
            return jsonify({'error': 'Error en el preprocesamiento del espectrograma.'}), 500

        print(f"✅ Espectrograma preprocesado: {processed_spectrogram.shape}")
        print(f"   Rango final: [{processed_spectrogram.min():.4f}, {processed_spectrogram.max():.4f}]")

        # PASO 3: Preparar para el modelo (añadir batch dimension)
        model_input = np.expand_dims(processed_spectrogram, axis=0)
        print(f"📊 Input para modelo: {model_input.shape}")

        # PASO 4: Predicción usando TensorFlow Lite
        print("🤖 Realizando predicción...")
        predictions = predict_with_tflite(model_input)
        
        if predictions is None:
            return jsonify({'error': 'Error durante la predicción del modelo.'}), 500
            
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_index])

        # Obtener emoción predicha
        predicted_emotion = emotion_labels[predicted_class_index]

        print(f"🎯 Predicción: {predicted_emotion} (confianza: {confidence:.2%})")

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
        print(f"❌ Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500
    
    finally:
        # Limpiar archivo temporal
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"🧹 Archivo temporal eliminado: {temp_audio_path}")
            except Exception as e:
                print(f"⚠️  Error al eliminar archivo temporal: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servidor y modelo."""
    status = {
        'server': 'running',
        'model_loaded': interpreter is not None,
        'labels_loaded': emotion_labels is not None,
        'available_emotions': list(emotion_labels) if emotion_labels is not None else [],
        'model_input_shape': str(input_details[0]['shape']) if input_details is not None else None,
        'model_type': 'TensorFlow Lite'
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

if __name__ == '__main__':
    print("🚀 Iniciando EmoClassifier...")
        
    if interpreter is None or emotion_labels is None:
        print("❌ ERROR: No se pudo cargar el modelo o las etiquetas.")
        print("Verifica que los archivos existan en las rutas especificadas:")
        print(f"- Modelo: {MODEL_PATH}")
        print(f"- Labels: {LABELS_PATH}")
    else:
        print("✅ Modelo y etiquetas cargados correctamente.")
        print(f"✅ Emociones disponibles: {list(emotion_labels)}")
        print(f"✅ Input shape del modelo: {input_details[0]['shape']}")
    
    # Iniciar Flask
    app.run(debug=True, port=5000, host='0.0.0.0')