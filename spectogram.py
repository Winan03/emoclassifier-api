from utils.audio_to_image import audio_to_mel_spectrogram
import matplotlib.pyplot as plt

audio_file_path_tess = 'tess_data/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav'
spectrogram = audio_to_mel_spectrogram(
    audio_file_path_tess,
    img_size=(128, 128),
    sr=22050,
    duration=3.0,
    n_mels=128,
    n_fft=2048,
    fmax=8000
)

if spectrogram is not None:
    print(f"Forma: {spectrogram.shape}, Rango: [{spectrogram.min():.2f}, {spectrogram.max():.2f}]")
    plt.imsave('espectrograma_tess.png', spectrogram)
    print("Espectrograma guardado como 'espectrograma_tess.png'")
else:
    print("Error: No se pudo generar el espectrograma.")