import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg') # Asegura que Matplotlib no intente mostrar ventanas GUI
import matplotlib.pyplot as plt
import io
import cv2
import os
import soundfile as sf

def validate_audio_file(audio_file_path: str, sr: int = 22050, duration: float = 0.1) -> bool:
    """
    Valida que el archivo de audio sea válido, legible y compatible con tu pipeline.
    
    Args:
        audio_file_path (str): Ruta al archivo de audio.
        sr (int): Sample rate a usar (debe ser igual al del pipeline).
        duration (float): Duración mínima a intentar cargar (seg).
    Returns:
        bool: True si el archivo es válido, False si está corrupto o vacío.
    """
    try:
        # Validar extensión soportada
        valid_exts = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.webm') # Asegúrate de que .webm esté aquí
        ext = os.path.splitext(audio_file_path)[1].lower()
        if ext not in valid_exts:
            print(f"[validate_audio_file] Extensión no soportada: {audio_file_path}")
            return False

        # Validar que el archivo existe y tiene tamaño
        if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
            print(f"[validate_audio_file] Archivo no encontrado o vacío: {audio_file_path}")
            return False

        # Intentar cargar un fragmento corto (no todo el audio)
        # Aquí, 'sr' se usa para el librosa.load como lo tenías.
        y, _ = librosa.load(audio_file_path, sr=sr, mono=True, duration=duration)
        if y is None or len(y) == 0 or np.allclose(y, 0):
            print("[validate_audio_file] Audio vacío, corrupto o silencioso.")
            return False

        # Si todo salió bien, es válido
        return True

    except Exception as e:
        print(f"[validate_audio_file] Error al validar archivo: {e}")
        import traceback
        traceback.print_exc()
        return False

def audio_to_mel_spectrogram(audio_file_path: str,
                             sr: int = 22050,
                             duration: float = 3.0,
                             n_mels: int = 128,
                             n_fft: int = 2048,
                             fmax: int = 8000,
                             img_size: tuple = (128, 128)) -> np.ndarray:
    """
    Convierte un archivo de audio en un espectrograma Mel normalizado y lo
    devuelve como un array NumPy RGB (HxWx3) listo para VGG16, idéntico al usado en entrenamiento.
    Esta función NO incluye colormap, sino que apila el espectrograma en escala de grises.

    Args:
        audio_file_path (str): Ruta al archivo de audio (.wav, .mp3, etc.)
        sr (int): Tasa de muestreo para el audio.
        duration (float): Duración máxima a cargar (segundos).
        n_mels (int): Número de bandas Mel.
        n_fft (int): Tamaño de ventana FFT.
        fmax (int): Frecuencia máxima a incluir en el espectrograma Mel.
        img_size (tuple): Tamaño final (ancho, alto) del espectrograma.

    Returns:
        np.ndarray: Array (img_size[0], img_size[1], 3), valores [0,1], tipo float32, o None si error.
    """
    try:
        if not os.path.exists(audio_file_path):
            print(f"Error: El archivo {audio_file_path} no existe.")
            return None

        # Cargar el audio
        y, original_sr = librosa.load(audio_file_path, sr=sr, duration=duration)
        # Normalizar el volumen
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y)) * 0.9
        else: # Si el audio es silencio, llenarlo con ceros para evitar NaNs
            y = np.zeros_like(y)

        # Asegurar duración exacta
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        elif len(y) > target_length:
            y = y[:target_length]

        # Generar el espectrograma Mel con fmax
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, fmax=fmax
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalizar a [0, 1]
        mel_normalized = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        # Redimensionar a img_size
        mel_resized = cv2.resize(mel_normalized, img_size, interpolation=cv2.INTER_AREA)

        # Apilar en 3 canales idénticos (RGB)
        mel_rgb = np.stack([mel_resized]*3, axis=-1).astype(np.float32)

        return mel_rgb

    except Exception as e:
        print(f"Error al procesar '{audio_file_path}': {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_colored_spectrogram_image(mel_db_spectrogram: np.ndarray, img_size: tuple = (128, 128)) -> np.ndarray:
    """
    Genera una imagen RGB con colormap a partir de un espectrograma Mel en escala logarítmica (dB).
    Esta función es solo para visualización en el frontend.

    Args:
        mel_db_spectrogram (np.ndarray): El espectrograma Mel en escala de dB (salida de power_to_db).
        img_size (tuple): Tamaño deseado de la imagen de salida (ancho, alto).

    Returns:
        np.ndarray: Array (img_size[0], img_size[1], 3), valores [0,1], tipo float32, para visualización.
    """
    try:
        # Crear una figura de Matplotlib sin mostrarla (backend no interactivo)
        fig = plt.figure(figsize=(img_size[0]/100, img_size[1]/100), dpi=100) # Ajustar figsize/dpi
        ax = fig.add_subplot(111)
        ax.set_axis_off() # Eliminar ejes
        ax.pcolormesh(
            mel_db_spectrogram, 
            cmap='viridis', # Colormap para visualización. Puedes cambiarlo a 'plasma', 'magma', 'jet', etc.
            shading='gouraud'
        )
        # Ajustar los límites para que no haya espacio blanco extra
        ax.set_xlim([0, mel_db_spectrogram.shape[1]])
        ax.set_ylim([0, mel_db_spectrogram.shape[0]])
        plt.tight_layout(pad=0) # Eliminar todo el padding

        # Convertir la figura de Matplotlib a un array NumPy (imagen RGB)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_rgb = plt.imread(buf)
        plt.close(fig) # CERRAR la figura para liberar memoria

        # img_rgb estará en el rango [0, 1] y tendrá 4 canales (RGBA).
        # Eliminamos el canal alfa si solo necesitamos RGB
        if img_rgb.shape[2] == 4:
            img_rgb = img_rgb[:, :, :3]
        
        return img_rgb.astype(np.float32)

    except Exception as e:
        print(f"Error al generar imagen de espectrograma con color: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("--- Prueba de utils/audio_to_image.py ---")
    
    sample_audio_path = "sample_audio.wav"
    
    if not os.path.exists(sample_audio_path):
        print(f"Creando archivo de audio de prueba: {sample_audio_path}")
        try:
            samplerate = 22050
            duration = 3.0
            t = np.linspace(0, duration, int(samplerate * duration), False)
            
            # Crear un audio de prueba más complejo (mezcla de frecuencias)
            data = (0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
                    0.2 * np.sin(2 * np.pi * 880 * t) +  # A5
                    0.1 * np.sin(2 * np.pi * 220 * t) +  # A3
                    0.05 * np.random.normal(0, 1, len(t))) # Ruido
            
            # Aplicar envolvente para simular habla
            envelope = np.exp(-t * 0.5) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
            data = data * envelope
            
            sf.write(sample_audio_path, data, samplerate)
            print("Archivo de prueba creado exitosamente.")
            
        except Exception as e:
            print(f"Error al crear archivo de prueba: {e}")
    
    # Probar la función con audio real
    if os.path.exists(sample_audio_path):
        print(f"\n🎵 Procesando audio real: {sample_audio_path}")
        
        if validate_audio_file(sample_audio_path):
            print("✅ Archivo de audio válido")
            
            # Generar espectrograma para el modelo (escala de grises apilada)
            spectrogram_for_model = audio_to_mel_spectrogram(
                sample_audio_path, 
                img_size=(128, 128),
                sr=22050,
                duration=3.0
            )

            if spectrogram_for_model is not None:
                print(f"✅ Espectrograma para MODELO generado exitosamente!")
                print(f"   Forma: {spectrogram_for_model.shape}")
                print(f"   Tipo: {spectrogram_for_model.dtype}")
                print(f"   Rango: [{spectrogram_for_model.min():.4f}, {spectrogram_for_model.max():.4f}]")
                print(f"   Listo para VGG16 (espera 3 canales): {spectrogram_for_model.shape == (128, 128, 3)}")
                
                # Para generar el espectrograma VISUAL, necesitamos el espectrograma de Mel en dB
                # Lo calculamos aquí para la prueba, pero en app.py lo obtendrás de la primera etapa
                y_visual, _ = librosa.load(sample_audio_path, sr=22050, duration=3.0, mono=True)
                mel_spectrogram_visual = librosa.feature.melspectrogram(
                    y=y_visual, sr=22050, n_mels=128, n_fft=2048, fmax=8000
                )
                mel_db_visual = librosa.power_to_db(mel_spectrogram_visual, ref=np.max)

                colored_spectrogram_image = generate_colored_spectrogram_image(mel_db_visual, img_size=(128, 128))

                if colored_spectrogram_image is not None:
                    print(f"✅ Espectrograma COLOREADO para VISUALIZACIÓN generado exitosamente!")
                    print(f"   Forma: {colored_spectrogram_image.shape}")
                    print(f"   Tipo: {colored_spectrogram_image.dtype}")
                    print(f"   Rango: [{colored_spectrogram_image.min():.4f}, {colored_spectrogram_image.max():.4f}]")
                else:
                    print("❌ Error al generar el espectrograma COLOREADO")

            else:
                print("❌ Error al generar el espectrograma para el MODELO")
        else:
            print("❌ Archivo de audio no válido")
    else:
        print(f"❌ No se encontró archivo de audio: {sample_audio_path}")
        print("Coloca un archivo de audio válido (.wav, .mp3) en la misma carpeta para probar.")