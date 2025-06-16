// main.js - Lógica principal de la aplicación EmoClassifier y gestión de la interfaz de usuario

class EmoClassifierApp {
    constructor() {
        // Referencias a las secciones principales de la UI
        this.mainSection = document.getElementById('inicio');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');

        // Referencias a elementos dentro de la sección principal (Inicio)
        this.recordModeBtn = document.querySelector('.mode-btn[data-mode="record"]');
        this.uploadModeBtn = document.querySelector('.mode-btn[data-mode="upload"]');
        this.recordArea = document.getElementById('record-area');
        this.uploadArea = document.getElementById('upload-area');
        this.processBtn = document.getElementById('processBtn'); // Botón de procesar
        this.statusItems = document.querySelectorAll('.status-item'); // Indicadores de estado

        // Referencias a elementos dentro de la sección de resultados
        this.emotionIcon = document.getElementById('emotionIcon');
        this.emotionLabel = document.getElementById('emotionLabel');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.spectrogramImage = document.getElementById('spectrogramImage');
        this.audioPlayer = document.getElementById('audioPlayer');
        this.playBtn = document.getElementById('playBtn');
        this.progressBar = document.querySelector('.custom-audio-player .progress-bar'); // Barra de progreso del reproductor
        this.progressFill = document.querySelector('.custom-audio-player .progress-fill');
        this.timeDisplay = document.querySelector('.custom-audio-player .time-display');
        this.tryAgainBtn = document.getElementById('tryAgainBtn');
        this.shareBtn = document.getElementById('shareBtn');
        this.probabilityBarsContainer = document.getElementById('probabilityBars');

        // Referencias del header para desplazamiento suave y efecto
        this.navLinks = document.querySelectorAll('.nav-link');
        this.hamburger = document.querySelector('.hamburger');
        this.navMenu = document.querySelector('.nav-menu');
        this.header = document.querySelector('.header');

        // Opcional: Registra advertencias si los elementos críticos no se encuentran
        this.checkCriticalElements();

        this.init();
    }

    /**
     * Verifica si los elementos DOM críticos están presentes y registra advertencias si no lo están.
     */
    checkCriticalElements() {
        const criticalElements = {
            'mainSection': this.mainSection,
            'loadingSection': this.loadingSection,
            'resultsSection': this.resultsSection,
            'audioPlayer': this.audioPlayer,
            'header': this.header
        };

        for (const [name, element] of Object.entries(criticalElements)) {
            if (!element) {
                console.warn(`[EmoClassifierApp] Elemento DOM crítico no encontrado: #${name} o .${name}. Algunas funcionalidades pueden no operar.`);
            }
        }
    }

    /**
     * Inicializa la aplicación, configurando el estado inicial y los listeners.
     */
    init() {
        this.setupInitialState();
        this.addEventListeners();
        this.setupAudioPlayerEvents(); // Mover aquí para asegurar que los listeners del reproductor se configuren
        this.setupSmoothScrolling();
        this.setupHeaderScrollEffect();
        this.createCustomMessageModal(); // Inicializa el modal de mensajes personalizados
        this.setupModeSelector(); // Configura los selectores de modo (grabar/subir)
    }

    /**
     * Configura la visibilidad inicial de las secciones.
     */
    setupInitialState() {
        if (this.mainSection) this.mainSection.style.display = 'flex'; // La sección principal siempre visible inicialmente
        if (this.loadingSection) this.loadingSection.style.display = 'none';
        if (this.resultsSection) this.resultsSection.style.display = 'none';
        // Asegurarse de que el área de grabación esté activa por defecto
        if (this.recordArea) this.recordArea.classList.add('active');
        if (this.uploadArea) this.uploadArea.classList.remove('active');
        if (this.recordModeBtn) this.recordModeBtn.classList.add('active');
        if (this.uploadModeBtn) this.uploadModeBtn.classList.remove('active');
    }

    /**
     * Añade todos los event listeners necesarios para la interfaz de usuario.
     */
    addEventListeners() {
        // Los listeners del audioPlayer se manejan en setupAudioPlayerEvents()
        if (this.tryAgainBtn) {
            this.tryAgainBtn.addEventListener('click', () => this.resetApp());
        }
        if (this.shareBtn) {
            this.shareBtn.addEventListener('click', () => this.shareResults());
        }
        if (this.hamburger) {
            this.hamburger.addEventListener('click', () => {
                this.hamburger.classList.toggle('active');
                this.navMenu.classList.toggle('active');
            });
        }
        // Cerrar menú móvil al hacer clic en un enlace
        this.navLinks.forEach(link => {
            link.addEventListener('click', () => {
                if (this.navMenu && this.navMenu.classList.contains('active')) {
                    this.hamburger.classList.remove('active');
                    this.navMenu.classList.remove('active');
                }
            });
        });

        // Listener para el modal de espectrograma
        if (this.spectrogramImage) {
            this.spectrogramImage.addEventListener('click', () => {
                const modal = document.getElementById('spectrogramModal');
                const modalImg = document.getElementById('modalSpectrogramImage');
                if (modal && modalImg) {
                    modalImg.src = this.spectrogramImage.src;
                    // Llamar a la función animada expuesta por animations.js
                    if (window.showModalAnimated) {
                        window.showModalAnimated();
                    } else {
                        modal.classList.add('active'); // Fallback para mostrar
                    }
                }
            });
        }

        // Listener para el botón de procesar
        if (this.processBtn) {
            this.processBtn.addEventListener('click', async () => {
                console.log("Botón Analizar Emoción presionado.");
                this.showLoadingSection(); // Muestra la sección de carga
                if (window.audioRecorder) {
                    // Llama a la función de procesar del audio-recorder.js
                    await window.audioRecorder.processAudio();
                } else {
                    window.showCustomMessage('Error', 'El módulo de grabación/subida de audio no está cargado correctamente.', 'error');
                    this.resetApp(); // Vuelve al inicio si hay un error crítico
                }
            });
        }
    }

    /**
     * Configura los event listeners para el reproductor de audio de resultados.
     */
    setupAudioPlayerEvents() {
        if (!this.audioPlayer || !this.playBtn || !this.progressBar || !this.progressFill || !this.timeDisplay) {
            console.warn("[EmoClassifierApp] No se pueden configurar los eventos del reproductor de audio: faltan elementos DOM.");
            return;
        }

        this.playBtn.addEventListener('click', () => this.togglePlayPause());
        this.audioPlayer.addEventListener('timeupdate', () => this.updateProgressBar());
        this.audioPlayer.addEventListener('ended', () => this.resetPlayer());
        
        // Listener para buscar en la barra de progreso
        this.progressBar.addEventListener('click', (e) => this.seekAudio(e));

        // Listener para actualizar la duración cuando la metadata se carga
        this.audioPlayer.addEventListener('loadedmetadata', () => {
            this.updateProgressBar(); // Actualiza el tiempo y el progreso inicial
        });
    }

    /**
     * Configura el comportamiento de los botones para seleccionar modo (grabar/subir).
     */
    setupModeSelector() {
        if (this.recordModeBtn) {
            this.recordModeBtn.addEventListener('click', () => this.activateMode('record'));
        }
        if (this.uploadModeBtn) {
            this.uploadModeBtn.addEventListener('click', () => this.activateMode('upload'));
        }
    }

    /**
     * Activa el modo seleccionado (grabar o subir archivo).
     * @param {string} mode - 'record' o 'upload'.
     */
    activateMode(mode) {
        if (!this.recordArea || !this.uploadArea || !this.recordModeBtn || !this.uploadModeBtn) return;

        if (mode === 'record') {
            this.recordArea.classList.add('active');
            this.uploadArea.classList.remove('active');
            this.recordModeBtn.classList.add('active');
            this.uploadModeBtn.classList.remove('active');
            // Resetear el área de subida si se cambia a grabación
            if (window.audioRecorder) {
                window.audioRecorder.resetUploadArea();
            }
        } else if (mode === 'upload') {
            this.recordArea.classList.remove('active');
            this.uploadArea.classList.add('active');
            this.recordModeBtn.classList.remove('active');
            this.uploadModeBtn.classList.add('active');
            // Resetear el área de grabación si se cambia a subida
            if (window.audioRecorder) {
                window.audioRecorder.resetRecordArea();
            }
        }
        // Actualizar el estado del botón de procesar después de cambiar de modo
        if (window.audioRecorder) {
            // Se asume que audio-recorder.js maneja la habilitación/deshabilitación basándose en si hay audio
            // Por lo tanto, no forzamos 'false' aquí, sino que dejamos que audio-recorder.js lo decida
            window.audioRecorder.updateProcessButtonState(this.audioPlayer && this.audioPlayer.src); 
        }
    }

    /**
     * Alterna entre el estado de reproducción y pausa del audio.
     */
    togglePlayPause() {
        if (!this.audioPlayer || !this.playBtn) return;

        if (this.audioPlayer.paused) {
            this.audioPlayer.play();
            this.playBtn.innerHTML = '<i class="fas fa-pause"></i>';
        } else {
            this.audioPlayer.pause();
            this.playBtn.innerHTML = '<i class="fas fa-play"></i>';
        }
    }

    /**
     * Actualiza la barra de progreso y el tiempo de reproducción del audio.
     */
    updateProgressBar() {
        if (!this.audioPlayer || !this.progressFill || !this.timeDisplay) return;

        const duration = this.audioPlayer.duration;
        const currentTime = this.audioPlayer.currentTime;

        // Comprobar si la duración es un número válido y no cero
        if (isNaN(duration) || duration <= 0) {
            this.progressFill.style.width = '0%';
            this.timeDisplay.textContent = '0:00 / 0:00';
            return;
        }

        const progressPercent = (currentTime / duration) * 100;
        this.progressFill.style.width = `${progressPercent}%`;

        this.timeDisplay.textContent = `${this.formatTime(currentTime)} / ${this.formatTime(duration)}`;
    }

    /**
     * Permite buscar una posición específica en el audio al hacer clic en la barra de progreso.
     * @param {MouseEvent} e - Evento de clic.
     */
    seekAudio(e) {
        if (!this.progressBar || !this.audioPlayer) return;

        const progressBarRect = this.progressBar.getBoundingClientRect();
        const clickX = e.clientX - progressBarRect.left;
        const progressBarWidth = progressBarRect.width;
        const duration = this.audioPlayer.duration;

        if (isNaN(duration) || duration === 0) return;

        const seekTime = (clickX / progressBarWidth) * duration;
        this.audioPlayer.currentTime = seekTime;
    }

    /**
     * Reinicia el reproductor de audio a su estado inicial.
     */
    resetPlayer() {
        if (this.playBtn) this.playBtn.innerHTML = '<i class="fas fa-play"></i>';
        if (this.progressFill) this.progressFill.style.width = '0%';
        if (this.audioPlayer) this.audioPlayer.currentTime = 0;
        if (this.timeDisplay) this.timeDisplay.textContent = '0:00 / 0:00';
    }

    /**
     * Formatea el tiempo en segundos a un formato de minutos:segundos.
     * @param {number} seconds - Tiempo en segundos.
     * @returns {string} - Tiempo formateado (ej. "2:35").
     */
    formatTime(seconds) {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min}:${sec < 10 ? '0' : ''}${sec}`;
    }

    /**
     * Actualiza la sección de resultados con la emoción detectada, confianza y espectrograma.
     * @param {object} result - Objeto con los resultados de la emoción (emotion, confidence, spectrogram_data, all_probabilities).
     * @param {string} audioUrl - URL del blob de audio original.
     */
    updateResultsSection(result, audioUrl) {
        if (!this.emotionIcon || !this.emotionLabel || !this.confidenceFill ||
            !this.confidenceValue || !this.spectrogramImage || !this.audioPlayer || !this.probabilityBarsContainer) {
            console.error("No se pueden actualizar los resultados: faltan elementos DOM críticos.");
            return;
        }

        // Mapeo de emociones a iconos de FontAwesome y colores (puedes expandir esto)
        const emotionMap = {
            // Calma
            'calma_fuerte': { icon: 'fas fa-leaf', color: '#27ae60', display: 'Calma Fuerte' },
            'calma_normal': { icon: 'fas fa-leaf', color: '#2ecc71', display: 'Calma Normal' },
            
            // Disgusto
            'disgusto_fuerte': { icon: 'fas fa-frown-open', color: '#34495e', display: 'Disgusto Fuerte' },
            'disgusto_normal': { icon: 'fas fa-frown', color: '#7f8c8d', display: 'Disgusto Normal' },
            
            // Felicidad
            'felicidad_fuerte': { icon: 'fas fa-grin-hearts', color: '#f39c12', display: 'Felicidad Fuerte' },
            'felicidad_normal': { icon: 'fas fa-smile', color: '#e67e22', display: 'Felicidad Normal' },
            
            // Ira
            'ira_fuerte': { icon: 'fas fa-angry', color: '#c0392b', display: 'Ira Fuerte' },
            'ira_normal': { icon: 'fas fa-frown', color: '#e74c3c', display: 'Ira Normal' },
            
            // Miedo
            'miedo_fuerte': { icon: 'fas fa-dizzy', color: '#8e44ad', display: 'Miedo Fuerte' },
            'miedo_normal': { icon: 'fas fa-flushed', color: '#9b59b6', display: 'Miedo Normal' },
            
            // Neutral
            'neutral_normal': { icon: 'fas fa-meh', color: '#95a5a6', display: 'Neutral' },
            
            // Sorpresa
            'sorpresa_fuerte': { icon: 'fas fa-surprise', color: '#f1c40f', display: 'Sorpresa Fuerte' },
            'sorpresa_normal': { icon: 'fas fa-grin-squint', color: '#f4d03f', display: 'Sorpresa Normal' },
            
            // Tristeza
            'tristeza_fuerte': { icon: 'fas fa-sad-cry', color: '#2980b9', display: 'Tristeza Fuerte' },
            'tristeza_normal': { icon: 'fas fa-sad-tear', color: '#3498db', display: 'Tristeza Normal' },
            
            // Fallback para emociones no reconocidas
            'default': { icon: 'fas fa-question-circle', color: '#95a5a6', display: 'Desconocido' }
        }; 

        const detectedEmotionKey = result.emotion.toLowerCase(); // Asegura minúsculas para la búsqueda
        const emotionInfo = emotionMap[detectedEmotionKey] || { icon: 'fas fa-question-circle', color: '#95a5a6', display: 'Desconocido' }; // Default

        this.emotionIcon.innerHTML = `<i class="${emotionInfo.icon}" style="color: ${emotionInfo.color};"></i>`;
        this.emotionLabel.textContent = emotionInfo.display; // Usa el nombre de visualización
        this.confidenceValue.textContent = `${(result.confidence * 100).toFixed(1)}%`;
        this.confidenceFill.style.width = `${(result.confidence * 100).toFixed(1)}%`;
        this.confidenceFill.style.background = `linear-gradient(90deg, ${emotionInfo.color}, var(--accent-color))`; // Gradiente basado en emoción

        // Actualiza el espectrograma (asumiendo que result.spectrogram_data es una URL o Base64)
        this.spectrogramImage.src = result.spectrogram_data;

        // Carga el audio original en el reproductor de la sección de resultados
        if (this.audioPlayer) {
            this.audioPlayer.src = audioUrl;
            this.audioPlayer.load(); // Carga la metadata para obtener la duración
            this.resetPlayer(); // Resetea el botón de play y la barra de progreso visualmente
        }

        // Llenar el análisis detallado con las probabilidades
        this.probabilityBarsContainer.innerHTML = ''; // Limpiar barras anteriores
        if (result.all_probabilities && Array.isArray(result.all_probabilities)) {
            // Ordenar las emociones por probabilidad de mayor a menor
            const sortedProbabilities = result.all_probabilities.sort((a, b) => b.probability - a.probability);

            sortedProbabilities.forEach(prob => {
                const emotionDisplay = emotionMap[prob.emotion.toLowerCase()] ? emotionMap[prob.emotion.toLowerCase()].display : prob.emotion;
                const emotionColor = emotionMap[prob.emotion.toLowerCase()] ? emotionMap[prob.emotion.toLowerCase()].color : 'var(--secondary-color)'; // Color por defecto

                const probabilityItem = document.createElement('div');
                probabilityItem.classList.add('probability-item');
                probabilityItem.innerHTML = `
                    <span class="probability-label">${emotionDisplay}</span>
                    <div class="probability-bar-container">
                        <div class="probability-fill" style="width: ${(prob.probability * 100).toFixed(1)}%; background: linear-gradient(90deg, ${emotionColor}, var(--secondary-color));"></div>
                        <span class="probability-value">${(prob.probability * 100).toFixed(1)}%</span>
                    </div>
                `;
                this.probabilityBarsContainer.appendChild(probabilityItem);
            });
        }
    }

    /**
     * Resetea la aplicación a su estado inicial, volviendo a la sección de inicio.
     */
    resetApp() {
        this.showMainSection(); // Vuelve a mostrar la sección principal
        // Reinicia el reproductor de audio
        if (this.audioPlayer) {
            this.audioPlayer.pause();
            this.audioPlayer.src = ''; // Limpia la fuente del audio
        }
        this.resetPlayer(); // Este ya maneja null checks internamente

        // Si existe una instancia de AudioRecorder, reiníciala
        if (window.audioRecorder) {
            window.audioRecorder.resetUploadArea();
            window.audioRecorder.resetRecordArea(); // Asegura resetear el área de grabación también
            window.audioRecorder.updateProcessButtonState(false);
        }
        // Asegúrate de resetear el modo activo a "record" y limpiar el área de subida
        this.activateMode('record');
        this.probabilityBarsContainer.innerHTML = ''; // Limpiar las barras de probabilidad
        console.log("Aplicación reiniciada.");
    }

    /**
     * Simula la funcionalidad de compartir resultados (puede expandirse para compartir real).
     */
    shareResults() {
        if (!this.emotionLabel || !this.confidenceValue) {
            window.showCustomMessage('Error al Compartir', 'No se pudieron obtener los datos de emoción para compartir.', 'error');
            return;
        }
        const emotion = this.emotionLabel.textContent;
        const confidence = this.confidenceValue.textContent;
        const shareText = `¡Mi EmoClassifier ha detectado que mi voz expresa ${emotion} con una confianza del ${confidence}! Visita para probarlo: ${window.location.href} #EmoClassifier #IA #AnálisisDeVoz`;

        if (navigator.share) {
            navigator.share({
                title: 'Mis Resultados de EmoClassifier',
                text: shareText,
                url: window.location.href // O un enlace específico a los resultados
            }).then(() => {
                console.log('Resultados compartidos con éxito.');
                window.showCustomMessage('Compartido', '¡Resultados compartidos con éxito!', 'success');
            }).catch((error) => {
                console.error('Error al compartir:', error);
                window.showCustomMessage('Error al Compartir', 'No se pudo compartir. Intenta copiar y pegar.', 'error');
            });
        } else {
            // Fallback: copiar al portapapeles
            console.log('Web Share API no soportada. Copiando al portapapeles.');
            // Usar document.execCommand para compatibilidad en iframes (navigator.clipboard.writeText puede no funcionar)
            const textarea = document.createElement('textarea');
            textarea.value = shareText;
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                window.showCustomMessage('Copiado', 'Resultados copiados al portapapeles.', 'info');
            } catch (err) {
                console.error('Error al copiar:', err);
                window.showCustomMessage('Error', 'No se pudo copiar al portapapeles.', 'error');
            } finally {
                document.body.removeChild(textarea);
            }
        }
    }

    /**
     * Configura el desplazamiento suave (scroll-behavior) para los enlaces de navegación.
     */
    setupSmoothScrolling() {
        if (!this.navLinks) return;

        this.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href');
                const targetSection = document.querySelector(targetId);
                if (targetSection) {
                    // Cierra el menú de hamburguesa si está abierto
                    if (this.hamburger && this.hamburger.classList.contains('active')) {
                        this.hamburger.classList.remove('active');
                        this.navMenu.classList.remove('active');
                    }
                    targetSection.scrollIntoView({ behavior: 'smooth' });
                    // Actualiza la clase 'active' de los enlaces de navegación
                    this.navLinks.forEach(nav => nav.classList.remove('active'));
                    link.classList.add('active');
                }
            });
        });

        // Actualizar el link activo al hacer scroll
        window.addEventListener('scroll', () => this.updateActiveNavLink());
        window.addEventListener('load', () => this.updateActiveNavLink()); // Al cargar la página
    }

    /**
     * Actualiza la clase 'active' del enlace de navegación según la sección visible.
     */
    updateActiveNavLink() {
        if (!this.navLinks || !this.header) return;

        const sections = document.querySelectorAll('main section');
        let currentActive = null;
        // Obtener la posición actual del scroll, ajustando para la altura del encabezado
        const scrollY = window.scrollY + (this.header ? this.header.offsetHeight : 0) + 10; // Añadir un offset

        sections.forEach(section => {
            if (section) {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.offsetHeight;
                if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
                    currentActive = section.id;
                }
            }
        });

        this.navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href').substring(1) === currentActive);
        });
    }

    /**
     * Añade un efecto visual al header cuando se hace scroll.
     */
    setupHeaderScrollEffect() {
        if (!this.header) return;

        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) { // Cuando el scroll supera 50px
                this.header.classList.add('scrolled');
            } else {
                this.header.classList.remove('scrolled');
            }
        });
    }

    /**
     * Crea un modal personalizado para mostrar mensajes al usuario (en lugar de alert()).
     */
    createCustomMessageModal() {
        const modalId = 'customMessageModal';
        const modalHtml = `
            <div id="${modalId}" class="modal custom-message-modal">
                <div class="modal-content">
                    <span class="modal-close-custom">&times;</span>
                    <div class="message-icon"></div>
                    <h3 class="message-title"></h3>
                    <p class="message-text"></p>
                    <button class="modal-ok-btn">OK</button>
                </div>
            </div>
        `;

        if (!document.getElementById(modalId)) {
            document.body.insertAdjacentHTML('beforeend', modalHtml);
        }

        const modal = document.getElementById(modalId);
        if (!modal) {
            console.error("Error: El elemento del modal de mensaje personalizado no se encontró. No se puede inicializar.");
            return;
        }

        const closeBtn = modal.querySelector('.modal-close-custom');
        const okBtn = modal.querySelector('.modal-ok-btn');
        const iconElement = modal.querySelector('.message-icon');
        const titleElement = modal.querySelector('.message-title');
        const textElement = modal.querySelector('.message-text');

        if (closeBtn) closeBtn.addEventListener('click', () => modal.classList.remove('active'));
        if (okBtn) okBtn.addEventListener('click', () => modal.classList.remove('active'));
        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });

        // Inyectar estilos para el modal de mensaje si no existen
        const styleId = 'custom-message-modal-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = `
                .custom-message-modal {
                    backdrop-filter: blur(8px);
                    background-color: rgba(0, 0, 0, 0.6);
                    display: none; /* Asegurar que esté oculto por defecto */
                }
                .custom-message-modal.active {
                    display: flex; /* Mostrar cuando se activa */
                }
                .custom-message-modal .modal-content {
                    padding: 30px;
                    border-radius: 20px;
                    max-width: 400px;
                    text-align: center;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 15px;
                    background: var(--card-bg); /* Usar la variable del tema */
                    border: 1px solid var(--border-color);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                    color: var(--text-color-light); /* Asegura que el texto sea visible */
                }
                .custom-message-modal .message-icon {
                    font-size: 3rem;
                    color: var(--info-color); /* Default info color */
                }
                .custom-message-modal .message-icon.success { color: var(--success-color); }
                .custom-message-modal .message-icon.error { color: var(--error-color); }
                .custom-message-modal .message-icon.warning { color: var(--accent-color); }
                .custom-message-modal .message-icon.info { color: var(--info-color); }

                .custom-message-modal .message-title {
                    font-size: 1.8rem;
                    color: var(--text-color-light);
                    margin-bottom: 5px;
                }
                .custom-message-modal .message-text {
                    font-size: 1.1rem;
                    color: rgba(255, 255, 255, 0.8);
                    margin-bottom: 20px;
                }
                .custom-message-modal .modal-ok-btn {
                    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)); /* Usar variables del tema */
                    color: var(--text-color-light);
                    padding: 10px 25px;
                    border: none;
                    border-radius: 10px;
                    font-size: 1.1rem;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                .custom-message-modal .modal-ok-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }
                .custom-message-modal .modal-close-custom {
                    position: absolute;
                    top: 10px;
                    right: 20px;
                    font-size: 2rem;
                    color: var(--text-color-light); /* Usar color del tema */
                    cursor: pointer;
                    transition: color 0.3s ease;
                }
                .custom-message-modal .modal-close-custom:hover {
                    color: var(--accent-color); /* Usar color del tema */
                }
            `;
            document.head.appendChild(style);
        }

        // Función global para mostrar el mensaje
        window.showCustomMessage = (title, message, type = 'info') => {
            if (!modal || !iconElement || !titleElement || !textElement) {
                console.error("No se puede mostrar el mensaje personalizado: faltan elementos del modal.");
                console.log(`Mensaje: ${title} - ${message}`); // Fallback a console.log
                return;
            }

            // Resetear clases de icono y añadir la clase de tipo
            iconElement.className = 'message-icon'; // Limpia clases anteriores
            let iconClass = '';
            switch (type) {
                case 'success': iconClass = 'fas fa-check-circle success'; break;
                case 'error': iconClass = 'fas fa-times-circle error'; break;
                case 'warning': iconClass = 'fas fa-exclamation-triangle warning'; break;
                case 'info': iconClass = 'fas fa-info-circle info'; break;
                default: iconClass = 'fas fa-info-circle info'; break;
            }
            iconElement.classList.add(...iconClass.split(' ')); // Añade clases del icono y tipo

            titleElement.textContent = title;
            textElement.textContent = message;
            modal.classList.add('active'); // Muestra el modal con la clase 'active'
        };
    }

    /**
     * Muestra la sección principal y oculta las demás.
     */
    showMainSection() {
        if (this.mainSection) this.mainSection.style.display = 'flex';
        if (this.loadingSection) this.loadingSection.style.display = 'none';
        if (this.resultsSection) this.resultsSection.style.display = 'none';
        document.body.style.overflowY = 'auto';
    }

    /**
     * Muestra la sección de carga y oculta las demás.
     */
    showLoadingSection() {
        if (this.mainSection) this.mainSection.style.display = 'none';
        if (this.loadingSection) this.loadingSection.style.display = 'flex';
        if (this.resultsSection) this.resultsSection.style.display = 'none';
        document.body.style.overflowY = 'hidden';

        // Inicia la animación de los pasos de carga
        if (window.animateLoadingSteps) {
            window.animateLoadingSteps();
        }
    }

    /**
     * Muestra la sección de resultados y oculta las demás.
     */
    showResultsSection() {
        if (this.mainSection) this.mainSection.style.display = 'none';
        if (this.loadingSection) this.loadingSection.style.display = 'none';
        if (this.resultsSection) this.resultsSection.style.display = 'flex';
        document.body.style.overflowY = 'auto';
        // Desplázate al inicio de la sección de resultados para que sea visible
        if (this.resultsSection) {
            this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
}

// Inicializa la aplicación cuando el DOM esté completamente cargado.
document.addEventListener('DOMContentLoaded', () => {
    window.emoClassifierApp = new EmoClassifierApp();

    // Expone funciones clave al ámbito global para que audio-recorder.js y animations.js puedan acceder a ellas.
    // Esto es necesario porque son scripts separados.
    window.showMainSection = window.emoClassifierApp.showMainSection.bind(window.emoClassifierApp);
    window.showLoadingSection = window.emoClassifierApp.showLoadingSection.bind(window.emoClassifierApp);
    window.showResultsSection = window.emoClassifierApp.showResultsSection.bind(window.emoClassifierApp);
    window.updateResultsSection = window.emoClassifierApp.updateResultsSection.bind(window.emoClassifierApp);
});