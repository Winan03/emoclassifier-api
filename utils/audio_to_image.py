import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI - CRÍTICO para memoria
import matplotlib.pyplot as plt
import io
import cv2
import os
import soundfile as sf
import gc  # Para garbage collection manual
import warnings
warnings.filterwarnings('ignore')  # Suprimir warnings para reducir overhead

# Configurar matplotlib para usar mínima memoria
plt.rcParams['figure.max_open_warning'] = 0
plt.ioff()  # Desactivar modo interactivo

def validate_audio_file(audio_file_path: str, sr: int = 22050, duration: float = 0.1) -> bool:
    """
    Valida que el archivo de audio sea válido, legible y compatible con tu pipeline.
    OPTIMIZADO: Carga solo una muestra muy pequeña para validar.
    """
    try:
        # Validar extensión soportada
        valid_exts = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.webm')
        ext = os.path.splitext(audio_file_path)[1].lower()
        if ext not in valid_exts:
            return False

        # Validar que el archivo existe y tiene tamaño
        if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
            return False

        # Cargar SOLO una muestra muy pequeña (0.1 segundos) para validar
        y, _ = librosa.load(audio_file_path, sr=sr, mono=True, duration=duration)
        
        if y is None or len(y) == 0:
            return False

        # Limpiar memoria inmediatamente
        del y
        gc.collect()
        
        return True

    except Exception as e:
        print(f"[validate_audio_file] Error: {e}")
        return False

def audio_to_mel_spectrogram(audio_file_path: str,
                             sr: int = 22050,
                             duration: float = 3.0,
                             n_mels: int = 128,
                             n_fft: int = 2048,
                             fmax: int = 8000,
                             img_size: tuple = (128, 128)) -> np.ndarray:
    """
    VERSIÓN OPTIMIZADA PARA MEMORIA: Convierte audio a espectrograma Mel con gestión agresiva de memoria.
    """
    try:
        if not os.path.exists(audio_file_path):
            print(f"Error: El archivo {audio_file_path} no existe.")
            return None

        # OPTIMIZACIÓN 1: Cargar audio con parámetros optimizados
        y, _ = librosa.load(
            audio_file_path, 
            sr=sr, 
            duration=duration, 
            mono=True,
            res_type='kaiser_fast'  # Algoritmo más rápido y eficiente en memoria
        )
        
        # OPTIMIZACIÓN 2: Procesar en chunks si es muy largo
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant', constant_values=0)
        elif len(y) > target_length:
            y = y[:target_length]

        # OPTIMIZACIÓN 3: Normalizar de manera eficiente
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = np.multiply(y, 0.9 / max_val, out=y)  # In-place operation
        
        # OPTIMIZACIÓN 4: Generar espectrograma con configuración optimizada
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            fmax=fmax,
            hop_length=n_fft//4,  # Reducir resolución temporal para ahorrar memoria
            power=2.0
        )
        
        # Liberar memoria del audio original
        del y
        gc.collect()
        
        # OPTIMIZACIÓN 5: Conversión a dB eficiente
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        del mel_spectrogram  # Liberar memoria inmediatamente
        gc.collect()

        # OPTIMIZACIÓN 6: Normalización in-place
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        mel_range = mel_max - mel_min + 1e-8
        
        # Normalizar in-place para ahorrar memoria
        mel_db -= mel_min
        mel_db /= mel_range
        
        # OPTIMIZACIÓN 7: Resize eficiente
        mel_resized = cv2.resize(
            mel_db, 
            img_size, 
            interpolation=cv2.INTER_AREA
        )
        del mel_db  # Liberar memoria original
        gc.collect()

        # OPTIMIZACIÓN 8: Stack eficiente para 3 canales
        mel_rgb = np.empty((img_size[1], img_size[0], 3), dtype=np.float32)
        mel_rgb[:, :, 0] = mel_resized
        mel_rgb[:, :, 1] = mel_resized  
        mel_rgb[:, :, 2] = mel_resized
        
        del mel_resized  # Liberar memoria
        gc.collect()

        return mel_rgb

    except Exception as e:
        print(f"Error al procesar '{audio_file_path}': {e}")
        # Limpiar memoria en caso de error
        gc.collect()
        return None

def generate_colored_spectrogram_image(mel_db_spectrogram: np.ndarray, img_size: tuple = (128, 128)) -> np.ndarray:
    """
    VERSIÓN OPTIMIZADA: Genera imagen RGB con manejo agresivo de memoria.
    """
    fig = None
    try:
        # OPTIMIZACIÓN 1: Crear figura con tamaño mínimo
        fig_width = img_size[0] / 100.0
        fig_height = img_size[1] / 100.0
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        
        # OPTIMIZACIÓN 2: Usar configuración de memoria mínima
        ax.pcolormesh(
            mel_db_spectrogram, 
            cmap='viridis',
            shading='nearest',  # Más eficiente que 'gouraud'
            rasterized=True     # Reduce uso de memoria
        )
        
        ax.set_xlim([0, mel_db_spectrogram.shape[1]])
        ax.set_ylim([0, mel_db_spectrogram.shape[0]])
        
        # OPTIMIZACIÓN 3: Configuración minimalista
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        # OPTIMIZACIÓN 4: Convertir a imagen de manera eficiente
        buf = io.BytesIO()
        plt.savefig(
            buf, 
            format='png', 
            bbox_inches='tight', 
            pad_inches=0,
            dpi=100,
            facecolor='none',
            edgecolor='none'
        )
        buf.seek(0)
        
        # OPTIMIZACIÓN 5: Cargar imagen optimizada
        img_rgb = plt.imread(buf)
        buf.close()
        
        # Cerrar figura INMEDIATAMENTE
        plt.close(fig)
        fig = None
        
        # OPTIMIZACIÓN 6: Procesar imagen eficientemente
        if img_rgb.shape[2] == 4:  # Remover canal alfa si existe
            img_rgb = img_rgb[:, :, :3]
        
        # OPTIMIZACIÓN 7: Redimensionar si es necesario
        if img_rgb.shape[:2] != img_size[::-1]:  # img_size es (width, height), shape es (height, width)
            img_rgb = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_AREA)
        
        # Limpiar memoria
        gc.collect()
        
        return img_rgb.astype(np.float32)

    except Exception as e:
        print(f"Error al generar imagen de espectrograma: {e}")
        return None
    
    finally:
        # CRÍTICO: Asegurar que la figura se cierre
        if fig is not None:
            plt.close(fig)
        gc.collect()

# FUNCIÓN ADICIONAL: Limpieza de memoria global
def cleanup_memory():
    """Limpia memoria global - llamar después de cada procesamiento."""
    plt.close('all')  # Cerrar todas las figuras
    gc.collect()      # Forzar garbage collection

if __name__ == "__main__":
    print("--- Prueba OPTIMIZADA de utils/audio_to_image.py ---")
    
    sample_audio_path = "sample_audio.wav"
    
    if not os.path.exists(sample_audio_path):
        print(f"Creando archivo de audio de prueba: {sample_audio_path}")
        try:
            samplerate = 22050
            duration = 3.0
            t = np.linspace(0, duration, int(samplerate * duration), False)
            
            # Crear audio de prueba
            data = (0.3 * np.sin(2 * np.pi * 440 * t) +
                    0.2 * np.sin(2 * np.pi * 880 * t) +
                    0.1 * np.sin(2 * np.pi * 220 * t) +
                    0.05 * np.random.normal(0, 1, len(t)))
            
            envelope = np.exp(-t * 0.5) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
            data = data * envelope
            
            sf.write(sample_audio_path, data, samplerate)
            print("✅ Archivo de prueba creado.")
            
        except Exception as e:
            print(f"❌ Error al crear archivo de prueba: {e}")
    
    # Probar funciones optimizadas
    if os.path.exists(sample_audio_path):
        print(f"\n🎵 Procesando audio: {sample_audio_path}")
        
        if validate_audio_file(sample_audio_path):
            print("✅ Archivo válido")
            
            # Test espectrograma para modelo
            spectrogram_for_model = audio_to_mel_spectrogram(
                sample_audio_path, 
                img_size=(128, 128),
                sr=22050,
                duration=3.0
            )

            if spectrogram_for_model is not None:
                print(f"✅ Espectrograma para MODELO generado!")
                print(f"   Forma: {spectrogram_for_model.shape}")
                print(f"   Tipo: {spectrogram_for_model.dtype}")
                print(f"   Rango: [{spectrogram_for_model.min():.4f}, {spectrogram_for_model.max():.4f}]")
                
                # Limpiar memoria después de cada operación
                cleanup_memory()
                
            else:
                print("❌ Error al generar espectrograma para modelo")
        else:
            print("❌ Archivo no válido")
    else:
        print(f"❌ No se encontró archivo: {sample_audio_path}")
    
    # Limpieza final
    cleanup_memory()
    print("🧹 Limpieza de memoria completada.")