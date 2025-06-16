import cv2
import numpy as np

def resize_spectrogram(spectrogram_array: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    """
    Redimensiona una imagen de espectrograma a un tamaño objetivo.
    Optimizado para espectrogramas de audio real.

    Args:
        spectrogram_array (np.ndarray): El array NumPy que representa el espectrograma
                                        (ej. HxW, HxWxC).
        target_size (tuple): La tupla (ancho, alto) del tamaño deseado.

    Returns:
        np.ndarray: El array redimensionado del espectrograma.
    """
    try:
        print(f"Redimensionando espectrograma de {spectrogram_array.shape} a {target_size}")
        
        # Asegurarse de que el array tenga al menos 2 dimensiones para el redimensionamiento
        if spectrogram_array.ndim < 2:
            raise ValueError(f"El array del espectrograma debe tener al menos 2 dimensiones, recibido: {spectrogram_array.ndim}D")

        # OpenCV espera (width, height) para target_size, pero el array es (height, width, channels)
        # Por eso, el target_size se pasa directamente, ya que cv2.resize lo interpreta como (width, height)
        resized_image = cv2.resize(spectrogram_array, target_size, interpolation=cv2.INTER_AREA)

        print(f"Redimensionado exitoso: {resized_image.shape}")
        return resized_image
        
    except Exception as e:
        print(f"Error al redimensionar espectrograma: {e}")
        raise

def normalize_spectrogram(spectrogram_array: np.ndarray, method='minmax') -> np.ndarray:
    """
    Normaliza los valores de píxeles de un espectrograma.
    Mejorado para manejar espectrogramas de audio real con diferentes rangos.

    Args:
        spectrogram_array (np.ndarray): El array NumPy del espectrograma.
        method (str): Método de normalización:
                      - 'minmax': Normaliza a [0, 1] usando min-max
                      - 'zscore': Normalización z-score (media=0, std=1)
                      - 'robust': Normalización robusta usando percentiles (IQR)

    Returns:
        np.ndarray: El array normalizado del espectrograma.
    """
    try:
        print(f"Normalizando espectrograma con método: {method}")
        print(f"Forma original: {spectrogram_array.shape}")
        print(f"Rango original: [{spectrogram_array.min():.4f}, {spectrogram_array.max():.4f}]")
        
        # Convertir a float32 para la normalización y evitar errores de tipo
        normalized_image = spectrogram_array.astype(np.float32)
        
        # Evitar divisiones por cero o rangos nulos
        if np.all(normalized_image == normalized_image.flat[0]):
            print("Advertencia: Todos los valores son iguales, devolviendo array de ceros o 0.5.")
            # Dependiendo del método, podría ser 0, o 0.5 si se espera un rango [0,1]
            return np.zeros_like(normalized_image) if method != 'minmax' else np.ones_like(normalized_image) * 0.5
            
        if method == 'minmax':
            # Normalización Min-Max a [0, 1]
            min_val = np.min(normalized_image)
            max_val = np.max(normalized_image)
            
            if max_val == min_val: # Doble comprobación para seguridad
                normalized_image = np.ones_like(normalized_image) * 0.5 # Valor neutro
            else:
                normalized_image = (normalized_image - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            # Normalización Z-score
            mean_val = np.mean(normalized_image)
            std_val = np.std(normalized_image)
            
            if std_val == 0: # Doble comprobación
                normalized_image = np.zeros_like(normalized_image) # Opcional: manejar como valores muy pequeños/constantes
            else:
                normalized_image = (normalized_image - mean_val) / std_val
                # Reescalar a [0, 1] después de z-score para modelos que lo esperan
                # Esto es crucial si el modelo VGG16 fue entrenado con inputs en [0, 1]
                min_norm = np.min(normalized_image)
                max_norm = np.max(normalized_image)
                if max_norm != min_norm:
                    normalized_image = (normalized_image - min_norm) / (max_norm - min_norm)
                else: # Si después de z-score sigue siendo constante, maneja como antes
                    normalized_image = np.zeros_like(normalized_image)


        elif method == 'robust':
            # Normalización robusta usando percentiles (menos sensible a outliers)
            # Calcula el primer cuartil (Q1) y el tercer cuartil (Q3)
            q1, q3 = np.percentile(normalized_image, [25, 75])
            iqr = q3 - q1 # Rango intercuartílico

            if iqr == 0:
                print("Advertencia: IQR es 0, usando normalización min-max como fallback.")
                min_val = np.min(normalized_image)
                max_val = np.max(normalized_image)
                if max_val != min_val:
                    normalized_image = (normalized_image - min_val) / (max_val - min_val)
                else: # Si min-max también es constante
                    normalized_image = np.ones_like(normalized_image) * 0.5
            else:
                # Normaliza usando Q1 y IQR
                normalized_image = (normalized_image - q1) / iqr
                # Opcional: Recortar a un rango estándar si los valores aún son extremos
                # Por ejemplo, a 0-1, aunque la normalización robusta no garantiza este rango
                normalized_image = np.clip(normalized_image, 0, 1) # Clamping para VGG16

        else:
            raise ValueError(f"Método de normalización no soportado: {method}. Métodos válidos: 'minmax', 'zscore', 'robust'.")

        print(f"Rango después de normalización: [{normalized_image.min():.4f}, {normalized_image.max():.4f}]")
        return normalized_image
        
    except Exception as e:
        print(f"Error al normalizar espectrograma: {e}")
        raise

def validate_spectrogram_for_vgg16(spectrogram_array: np.ndarray) -> bool:
    """
    Valida que el espectrograma tenga el formato correcto para VGG16.
    
    Args:
        spectrogram_array (np.ndarray): Array del espectrograma a validar
        
    Returns:
        bool: True si es válido para VGG16, False en caso contrario
    """
    try:
        # VGG16 espera imágenes de forma (height, width, 3) o (batch, height, width, 3)
        # Aquí validamos la imagen individual (height, width, 3)
        if spectrogram_array is None:
            print("❌ Espectrograma es None.")
            return False

        if len(spectrogram_array.shape) != 3:
            print(f"❌ VGG16 requiere 3 dimensiones (HxWxC), recibido: {len(spectrogram_array.shape)}D.")
            return False
            
        if spectrogram_array.shape[2] != 3:
            print(f"❌ VGG16 requiere 3 canales (RGB), recibido: {spectrogram_array.shape[2]} canales.")
            return False
            
        # Verificar que los valores estén en rango [0, 1] para modelos pre-entrenados
        # Una ligera tolerancia para errores de punto flotante
        if spectrogram_array.min() < -0.001 or spectrogram_array.max() > 1.001:
            print(f"⚠️  Advertencia: Valores fuera del rango [0,1]: [{spectrogram_array.min():.4f}, {spectrogram_array.max():.4f}]")
            print("   Esto podría afectar el rendimiento del modelo VGG16 pre-entrenado.")
            
        # Verificar que el array no esté vacío o con valores completamente constantes
        if spectrogram_array.size == 0 or np.all(spectrogram_array == spectrogram_array.flat[0]):
            print("⚠️  Advertencia: Espectrograma vacío o todos los valores son iguales.")
            print("   Esto sugiere un problema en el procesamiento del audio o un audio silente.")
            
        print(f"✅ Espectrograma con formato válido para VGG16: {spectrogram_array.shape}")
        return True
        
    except Exception as e:
        print(f"Error al validar espectrograma: {e}")
        return False

def preprocess_for_model(spectrogram_array: np.ndarray, target_size=(128, 128), 
                         normalize_method='minmax') -> np.ndarray:
    """
    Pipeline completo de preprocesamiento para espectrogramas de audio real.
    
    Args:
        spectrogram_array (np.ndarray): Espectrograma original (generado por audio_to_mel_spectrogram)
        target_size (tuple): Tamaño objetivo (ancho, alto) para el redimensionamiento.
        normalize_method (str): Método de normalización ('minmax', 'zscore', 'robust').
        
    Returns:
        np.ndarray: Espectrograma preprocesado listo para la inferencia del modelo.
    """
    try:
        print("=== Iniciando pipeline de preprocesamiento ===")
        
        # 1. Validación inicial del input
        if spectrogram_array is None:
            raise ValueError("El espectrograma de entrada es None. No se puede preprocesar.")
            
        print(f"Input original shape: {spectrogram_array.shape}")
        
        # 2. Redimensionar si es necesario
        # spectrogram_array de audio_to_mel_spectrogram ya tiene el tamaño y 3 canales
        # Si deseas un tamaño diferente, esta función lo aplicará.
        # Asumimos que audio_to_mel_spectrogram ya entrega (height, width, 3) y el tamaño correcto.
        # Esta llamada sería redundante si img_size en audio_to_mel_spectrogram es igual a target_size aquí.
        # Pero se mantiene por flexibilidad.
        current_shape_h, current_shape_w = spectrogram_array.shape[0], spectrogram_array.shape[1]
        if (current_shape_w, current_shape_h) != target_size: # cv2.resize espera (width, height)
            spectrogram_array = resize_spectrogram(spectrogram_array, target_size)
        else:
            print(f"Tamaño ya es correcto: {spectrogram_array.shape[:2]} (HxW).")
            
        # 3. Normalizar
        spectrogram_array = normalize_spectrogram(spectrogram_array, method=normalize_method)
        
        # 4. Validar para VGG16 (y otros modelos CNN que esperan 3 canales y rango [0,1])
        if validate_spectrogram_for_vgg16(spectrogram_array):
            print("✅ Preprocesamiento completado exitosamente.")
            return spectrogram_array
        else:
            raise ValueError("El espectrograma no pasó la validación final para el modelo (VGG16).")
            
    except Exception as e:
        print(f"❌ Error en el pipeline de preprocesamiento: {e}")
        raise # Re-lanza la excepción para ser manejada en el nivel superior

if __name__ == "__main__":
    print("=== Prueba de preprocess.py con datos realistas ===")

    # Simular un espectrograma más realista (como el que vendría de audio_to_mel_spectrogram)
    print("\n1. Simulando espectrograma realista (HxWxC, float32)...")
    
    # Crear datos que simulen un espectrograma real de mel con 3 canales (RGB)
    # Valores típicos después de normalización a [0,1] en audio_to_mel_spectrogram
    simulated_spectrogram_input = np.random.uniform(0.0, 1.0, size=(128, 128, 3)).astype(np.float32)
    
    # Añadir algunas características que podrían hacer los valores no uniformes
    # por ejemplo, simular una señal o ruido para asegurar que min/max no sean triviales
    simulated_spectrogram_input[20:40, 30:50, :] = np.random.uniform(0.7, 0.9, size=(20, 20, 3))
    simulated_spectrogram_input[80:100, 70:90, :] = np.random.uniform(0.1, 0.3, size=(20, 20, 3))
    
    print(f"Espectrograma realista simulado: {simulated_spectrogram_input.shape}")
    print(f"Rango de valores: [{simulated_spectrogram_input.min():.4f}, {simulated_spectrogram_input.max():.4f}]")

    # 2. Probar pipeline completo con 'minmax' (ya debería estar en [0,1])
    print("\n2. Probando pipeline completo con normalización 'minmax'...")
    try:
        processed_spec_minmax = preprocess_for_model(
            simulated_spectrogram_input, 
            target_size=(128, 128),
            normalize_method='minmax'
        )
        
        print(f"✅ Pipeline exitoso con 'minmax'!")
        print(f"   Forma final: {processed_spec_minmax.shape}")
        print(f"   Tipo: {processed_spec_minmax.dtype}")
        print(f"   Rango final: [{processed_spec_minmax.min():.4f}, {processed_spec_minmax.max():.4f}]")
        
        # Simular que se pasa al modelo (añadir dimensión de batch)
        model_input_minmax = np.expand_dims(processed_spec_minmax, axis=0)
        print(f"   Listo para modelo (con batch dimension): {model_input_minmax.shape}")
        
    except Exception as e:
        print(f"❌ Error en pipeline con 'minmax': {e}")
        import traceback
        traceback.print_exc()

    # 3. Probar pipeline completo con 'zscore'
    print("\n3. Probando pipeline completo con normalización 'zscore'...")
    try:
        # Usar una copia para no modificar el original de la prueba anterior
        processed_spec_zscore = preprocess_for_model(
            simulated_spectrogram_input.copy(), 
            target_size=(128, 128),
            normalize_method='zscore'
        )
        
        print(f"✅ Pipeline exitoso con 'zscore'!")
        print(f"   Forma final: {processed_spec_zscore.shape}")
        print(f"   Tipo: {processed_spec_zscore.dtype}")
        print(f"   Rango final: [{processed_spec_zscore.min():.4f}, {processed_spec_zscore.max():.4f}]")
        
        model_input_zscore = np.expand_dims(processed_spec_zscore, axis=0)
        print(f"   Listo para modelo (con batch dimension): {model_input_zscore.shape}")
        
    except Exception as e:
        print(f"❌ Error en pipeline con 'zscore': {e}")
        import traceback
        traceback.print_exc()

    # 4. Probar pipeline completo con 'robust'
    print("\n4. Probando pipeline completo con normalización 'robust'...")
    try:
        processed_spec_robust = preprocess_for_model(
            simulated_spectrogram_input.copy(), 
            target_size=(128, 128),
            normalize_method='robust'
        )
        
        print(f"✅ Pipeline exitoso con 'robust'!")
        print(f"   Forma final: {processed_spec_robust.shape}")
        print(f"   Tipo: {processed_spec_robust.dtype}")
        print(f"   Rango final: [{processed_spec_robust.min():.4f}, {processed_spec_robust.max():.4f}]")
        
        model_input_robust = np.expand_dims(processed_spec_robust, axis=0)
        print(f"   Listo para modelo (con batch dimension): {model_input_robust.shape}")
        
    except Exception as e:
        print(f"❌ Error en pipeline con 'robust': {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Fin de las pruebas de preprocess.py ===")
    print("💡 Tu espectrograma está listo para ser usado con VGG16!")