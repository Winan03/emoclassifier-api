// audio-recorder.js - Lógica para la grabación y subida de audio

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioBlob = null; // Almacena el Blob de audio grabado o subido
        this.audioUrl = null; // URL del Blob para previsualización
        this.isRecording = false;
        this.recordStartTime = null;
        this.recordDuration = 0; // Duración en milisegundos
        this.MAX_RECORD_DURATION = 15000; // 15 segundos en milisegundos para grabación
        this.countdownInterval = null;
        this.audioContext = null; // Contexto de audio para el visualizador y decodificación/codificación
        this.analyser = null; // Analizador para obtener datos de audio
        this.sourceNode = null; // Nodo de fuente del MediaStream para desconexión
        this.visualizerCanvas = document.getElementById('visualizerCanvas'); // Canvas del visualizador
        this.canvasCtx = this.visualizerCanvas ? this.visualizerCanvas.getContext('2d') : null;
        this.animationFrameId = null;

        // Referencias a elementos del DOM
        this.recordBtn = document.getElementById('recordBtn');
        this.recordStatusSpan = document.getElementById('recordStatus'); // Span para el estado de grabación
        this.recordTimerDiv = document.getElementById('recordTimer'); // Div para el temporizador
        this.audioVisualizerDiv = document.getElementById('audioVisualizer'); // Contenedor del visualizador
        this.recordControlsDiv = document.getElementById('recordControls'); // Controles después de grabar

        this.playRecordedBtn = document.getElementById('playRecordedBtn');
        this.pauseRecordedBtn = document.getElementById('pauseRecordedBtn');
        this.stopRecordedBtn = document.getElementById('stopRecordedBtn');
        this.deleteRecordedBtn = document.getElementById('deleteRecordedBtn');

        this.uploadZone = document.getElementById('uploadZone'); // Zona de arrastre y click
        this.audioFileInput = document.getElementById('audioFileInput'); // Input de tipo file
        this.uploadedFileInfoDiv = document.getElementById('uploadedFileInfo'); // Información del archivo subido
        this.fileNameSpan = document.getElementById('fileName');
        this.fileSizeSpan = document.getElementById('fileSize');
        this.removeFileBtn = document.getElementById('removeFileBtn');

        this.processBtn = document.getElementById('processBtn'); // Botón de procesar en la sección principal
        this.statusMicIcon = document.querySelector('.status-item .fa-microphone-alt'); // Icono de micrófono en estado

        // Referencia al elemento de audio principal para reproducción
        this.audioPlayerElement = document.getElementById('audioPlayer');

        this.init();
    }

    /**
     * Inicializa el grabador de audio, configurando eventos y permisos.
     */
    init() {
        this.addEventListeners();
        this.checkMicrophonePermission();
        this.resetRecordArea(); // Asegura el estado inicial del área de grabación
        this.resetUploadArea(); // Asegura el estado inicial del área de subida
        this.updateProcessButtonState(false); // Botón de procesar deshabilitado inicialmente
        this.resizeVisualizerCanvas(); // Ajusta el tamaño del canvas al iniciar
        window.addEventListener('resize', () => this.resizeVisualizerCanvas()); // Reajusta en redimensionamiento
    }

    /**
     * Ajusta el tamaño del canvas del visualizador para ser responsivo.
     */
    resizeVisualizerCanvas() {
        if (this.visualizerCanvas && this.audioVisualizerDiv) {
            this.visualizerCanvas.width = this.audioVisualizerDiv.offsetWidth;
            this.visualizerCanvas.height = this.audioVisualizerDiv.offsetHeight;
        }
    }

    /**
     * Solicita permiso para acceder al micrófono y actualiza el estado.
     */
    async checkMicrophonePermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // Si el permiso se concede, detenemos las pistas para que el icono del micro no quede activo en el navegador
            stream.getTracks().forEach(track => track.stop());
            this.recordBtn.disabled = false; // Habilita el botón de grabar
            this.recordBtn.classList.remove('disabled');
            if (this.recordStatusSpan) this.recordStatusSpan.textContent = "Listo para grabar";
            if (this.statusMicIcon) this.statusMicIcon.style.color = 'var(--accent-color)';
            console.log("Permiso de micrófono concedido.");
        } catch (error) {
            console.error("Permiso de micrófono denegado:", error);
            this.recordBtn.disabled = true; // Deshabilita el botón de grabar
            this.recordBtn.classList.add('disabled');
            if (this.recordStatusSpan) this.recordStatusSpan.textContent = "Micrófono no disponible";
            if (this.statusMicIcon) this.statusMicIcon.style.color = '#dc3545';
            if (window.showCustomMessage) {
                window.showCustomMessage('Permiso de Micrófono Denegado',
                                         'Por favor, habilita el acceso a tu micrófono en la configuración del navegador para usar la función de grabación.',
                                         'error');
            }
        }
    }

    /**
     * Añade todos los event listeners necesarios para la interacción del usuario.
     */
    addEventListeners() {
        if (this.recordBtn) {
            this.recordBtn.addEventListener('mousedown', (e) => this.startRecording(e));
            this.recordBtn.addEventListener('mouseup', () => this.stopRecording());
            this.recordBtn.addEventListener('mouseleave', () => {
                if (this.isRecording) this.stopRecording();
            });
            this.recordBtn.addEventListener('touchstart', (e) => {
                e.preventDefault();
                this.startRecording(e);
            }, { passive: false });
            this.recordBtn.addEventListener('touchend', () => this.stopRecording());
        }

        if (this.playRecordedBtn) this.playRecordedBtn.addEventListener('click', () => this.playRecordedAudio());
        if (this.pauseRecordedBtn) this.pauseRecordedBtn.addEventListener('click', () => this.pauseRecordedAudio());
        if (this.stopRecordedBtn) this.stopRecordedBtn.addEventListener('click', () => this.stopRecordedAudio());
        if (this.deleteRecordedBtn) this.deleteRecordedBtn.addEventListener('click', () => this.resetRecordArea());

        if (this.uploadZone) {
            this.uploadZone.addEventListener('click', () => this.audioFileInput.click());
            this.uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
            this.uploadZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            this.uploadZone.addEventListener('drop', (e) => this.handleDrop(e));
        }

        if (this.audioFileInput) {
            this.audioFileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        }

        if (this.removeFileBtn) {
            this.removeFileBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.resetUploadArea();
            });
        }
    }

    /**
     * Inicia la grabación de audio desde el micrófono.
     * @param {Event} e - El evento que disparó la grabación (ej. MouseDown, TouchStart).
     */
    async startRecording(e) {
        if (this.recordBtn.disabled || this.isRecording) {
            return;
        }

        this.resetRecordArea();

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.sourceNode = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.sourceNode.connect(this.analyser);
            this.analyser.fftSize = 2048;
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);

            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = async () => {
                // El Blob inicial del grabador (webm/opus)
                const recordedBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
                
                // Intentar convertir a WAV antes de usarlo o procesarlo
                const wavBlob = await this.convertToWav(recordedBlob);

                if (wavBlob) {
                    this.audioBlob = wavBlob; // Almacenamos el Blob WAV
                    this.audioUrl = URL.createObjectURL(this.audioBlob);
                    console.log("Grabación detenida. URL del audio (WAV):", this.audioUrl);
                } else {
                    console.error("Error al convertir a WAV. Usando el Blob original WebM.");
                    this.audioBlob = recordedBlob; // Fallback al original si la conversión falla
                    this.audioUrl = URL.createObjectURL(this.audioBlob);
                    console.log("Grabación detenida. URL del audio (WebM fallback):", this.audioUrl);
                }
            
                if (this.audioPlayerElement) {
                    this.audioPlayerElement.src = this.audioUrl;
                    this.audioPlayerElement.load();
                }
            
                stream.getTracks().forEach(track => track.stop());
                this.resetVisualizer();
                if (this.sourceNode) {
                    this.sourceNode.disconnect();
                    this.sourceNode = null;
                }
                if (this.audioContext) {
                    this.audioContext.close();
                    this.audioContext = null;
                }
                
                this.showRecordControls();
                this.updateProcessButtonState(true);
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.recordBtn.classList.add('recording');
            this.recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
            if (this.recordStatusSpan) this.recordStatusSpan.textContent = "Grabando...";
            this.recordStartTime = Date.now();
            this.startCountdown();
            this.drawVisualizer();
            console.log("Grabación iniciada.");
        } catch (error) {
            console.error("Error al iniciar la grabación:", error);
            if (window.showCustomMessage) {
                window.showCustomMessage('Error de Grabación',
                                         'No se pudo iniciar la grabación. Asegúrate de que el micrófono esté disponible y los permisos concedidos.',
                                         'error');
            }
            this.recordBtn.disabled = true;
            this.recordBtn.classList.add('disabled');
            if (this.audioContext) {
                this.audioContext.close();
                this.audioContext = null;
            }
        }
    }

    /**
     * Detiene la grabación de audio.
     */
    stopRecording() {
        if (!this.isRecording) {
            return;
        }
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        this.isRecording = false;
        this.recordBtn.classList.remove('recording');
        this.recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        if (this.recordStatusSpan) this.recordStatusSpan.textContent = "Presiona para grabar";
        clearInterval(this.countdownInterval);
        cancelAnimationFrame(this.animationFrameId);
        this.recordDuration = 0;
        if (this.recordTimerDiv) this.recordTimerDiv.textContent = '0:00';
        console.log("Grabación detenida.");
        this.updateProcessButtonState(this.audioBlob !== null);
    }

    /**
     * Inicia la cuenta regresiva para la duración máxima de grabación.
     */
    startCountdown() {
        clearInterval(this.countdownInterval);
        if (this.recordTimerDiv) this.recordTimerDiv.textContent = this.formatTime(0);

        this.countdownInterval = setInterval(() => {
            if (!this.isRecording) {
                clearInterval(this.countdownInterval);
                return;
            }
            this.recordDuration = Date.now() - this.recordStartTime;
            if (this.recordTimerDiv) this.recordTimerDiv.textContent = this.formatTime(this.recordDuration);

            if (this.recordDuration >= this.MAX_RECORD_DURATION) {
                this.stopRecording();
                if (window.showCustomMessage) {
                    window.showCustomMessage('Grabación Completada', 'Se ha alcanzado la duración máxima de grabación (15 segundos).', 'info');
                }
            }
        }, 1000);
    }

    /**
     * Dibuja las ondas de audio en el canvas del visualizador.
     */
    drawVisualizer() {
        if (!this.canvasCtx || !this.analyser || !this.visualizerCanvas || !this.isRecording) {
            cancelAnimationFrame(this.animationFrameId);
            return;
        }

        this.animationFrameId = requestAnimationFrame(() => this.drawVisualizer());

        this.analyser.getByteTimeDomainData(this.dataArray);

        this.canvasCtx.fillStyle = 'rgba(0, 0, 0, 0)';
        this.canvasCtx.clearRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);

        this.canvasCtx.lineWidth = 2;
        this.canvasCtx.strokeStyle = 'var(--accent-color)';
        this.canvasCtx.beginPath();

        const sliceWidth = this.visualizerCanvas.width * 1.0 / this.dataArray.length;
        let x = 0;

        for (let i = 0; i < this.dataArray.length; i++) {
            const v = this.dataArray[i] / 128.0;
            const y = v * this.visualizerCanvas.height / 2;

            if (i === 0) {
                this.canvasCtx.moveTo(x, y);
            } else {
                this.canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        this.canvasCtx.lineTo(this.visualizerCanvas.width, this.visualizerCanvas.height / 2);
        this.canvasCtx.stroke();
    }

    /**
     * Reinicia el visualizador, deteniendo la animación y limpiando el canvas.
     */
    resetVisualizer() {
        cancelAnimationFrame(this.animationFrameId);
        if (this.canvasCtx && this.visualizerCanvas) {
            this.canvasCtx.clearRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);
        }
    }

    /**
     * Muestra los controles de reproducción/borrado del audio grabado y oculta el botón de grabar.
     */
    showRecordControls() {
        if (this.recordControlsDiv) {
            this.recordControlsDiv.classList.remove('hidden');
        }
        if (this.recordBtn) {
            this.recordBtn.classList.add('hidden');
        }
        if (this.recordStatusSpan) this.recordStatusSpan.textContent = "Audio grabado";
        if (this.recordTimerDiv) this.recordTimerDiv.textContent = this.formatTime(this.recordDuration);
    }

    /**
     * Oculta los controles de grabación y muestra el botón de grabar.
     */
    hideRecordControls() {
        if (this.recordControlsDiv) {
            this.recordControlsDiv.classList.add('hidden');
        }
        if (this.recordBtn) {
            this.recordBtn.classList.remove('hidden');
        }
    }

    /**
     * Reproduce el audio grabado.
     */
    playRecordedAudio() {
        if (!this.audioUrl) {
            console.error("No hay audio grabado para reproducir.");
            if(window.showCustomMessage) {
                window.showCustomMessage('Error de Reproducción', 'No hay audio grabado para reproducir.', 'error');
            }
            return;
        }

        if (this.audioPlayerElement) {
            this.audioPlayerElement.src = this.audioUrl;
            this.audioPlayerElement.load();
            this.audioPlayerElement.play();
            this.playRecordedBtn.classList.add('hidden');
            this.pauseRecordedBtn.classList.remove('hidden');

            this.audioPlayerElement.onended = () => {
                this.playRecordedBtn.classList.remove('hidden');
                this.pauseRecordedBtn.classList.add('hidden');
            };
            this.audioPlayerElement.onpause = () => {
                this.playRecordedBtn.classList.remove('hidden');
                this.pauseRecordedBtn.classList.add('hidden');
            };
        }
    }

    /**
     * Pausa el audio grabado.
     */
    pauseRecordedAudio() {
        if (this.audioPlayerElement && !this.audioPlayerElement.paused) {
            this.audioPlayerElement.pause();
            this.playRecordedBtn.classList.remove('hidden');
            this.pauseRecordedBtn.classList.add('hidden');
        }
    }

    /**
     * Detiene el audio grabado y lo reinicia.
     */
    stopRecordedAudio() {
        if (this.audioPlayerElement) {
            this.audioPlayerElement.pause();
            this.audioPlayerElement.currentTime = 0;
            this.playRecordedBtn.classList.remove('hidden');
            this.pauseRecordedBtn.classList.add('hidden');
        }
    }

    /**
     * Formatea el tiempo en milisegundos a un formato de minutos:segundos.
     * @param {number} ms - Tiempo en milisegundos.
     * @returns {string} - Tiempo formateado (ej. "0:15").
     */
    formatTime(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
    }

    /**
     * Habilita o deshabilita el botón de procesar audio.
     * @param {boolean} enable - True para habilitar, False para deshabilitar.
     */
    updateProcessButtonState(enable = false) {
        if (this.processBtn) {
            const hasAudio = (this.audioBlob && this.audioBlob.size > 0) || (this.audioFileInput && this.audioFileInput.files.length > 0);
            if (enable && hasAudio) {
                this.processBtn.disabled = false;
                this.processBtn.classList.remove('disabled');
            } else {
                this.processBtn.disabled = true;
                this.processBtn.classList.add('disabled');
            }
        }
    }

    /**
     * Reinicia el área de grabación a su estado inicial.
     * También revoca la URL del Blob del audio grabado anteriormente.
     */
    resetRecordArea() {
        this.stopRecordedAudio();
        this.revokeAudioUrl();

        this.audioBlob = null;
        this.isRecording = false;
        clearInterval(this.countdownInterval);
        cancelAnimationFrame(this.animationFrameId);

        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            const stream = this.mediaRecorder.stream;
            stream.getTracks().forEach(track => track.stop());
        }

        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.recordBtn) {
            this.recordBtn.classList.remove('recording');
            this.recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            this.recordBtn.disabled = false;
            this.recordBtn.classList.remove('disabled');
        }

        if (this.recordStatusSpan) this.recordStatusSpan.textContent = "Presiona para grabar";
        if (this.recordTimerDiv) this.recordTimerDiv.textContent = "0:00";

        this.resetVisualizer();
        this.hideRecordControls(); 

        this.updateProcessButtonState(false);
        this.checkMicrophonePermission();
    }

    /**
     * Revoca la URL del Blob cuando ya no la necesitas.
     * Este método se llama internamente por resetRecordArea y resetUploadArea.
     */
    revokeAudioUrl() {
        if (this.audioUrl) {
            URL.revokeObjectURL(this.audioUrl);
            this.audioUrl = null;
            console.log("Recurso de audio liberado.");
        }
    }

    /**
     * Maneja el evento de arrastrar sobre el área de subida.
     * @param {DragEvent} e - Evento de arrastre.
     */
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        if (this.uploadZone) this.uploadZone.classList.add('drag-over');
    }

    /**
     * Maneja el evento de salir del área de subida.
     * @param {DragEvent} e - Evento de arrastre.
     */
    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        if (this.uploadZone) this.uploadZone.classList.remove('drag-over');
    }

    /**
     * Maneja el evento de soltar un archivo en el área de subida.
     * @param {DragEvent} e - Evento de soltar.
     */
    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        if (this.uploadZone) this.uploadZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelect(files[0]);
        } else {
            if (window.showCustomMessage) {
                window.showCustomMessage('Error de Archivo', 'Por favor, suelta solo un archivo de audio.', 'warning');
            }
        }
    }

    /**
     * Maneja la selección de un archivo de audio (ya sea por input o arrastrar y soltar).
     * Decodifica y recodifica el audio a WAV para asegurar compatibilidad.
     * @param {File} file - El archivo de audio seleccionado.
     */
    async handleFileSelect(file) {
        if (!file) {
            this.resetUploadArea();
            return;
        }

        this.revokeAudioUrl(); // Revocar cualquier URL anterior antes de crear una nueva

        // Verifica el tipo de archivo (solo si es uno de los aceptados por el MediaRecorder o comunes)
        const acceptedInputTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/mp4', 'audio/ogg', 'audio/webm'];
        if (!acceptedInputTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.wav') && !file.name.toLowerCase().endsWith('.mp3')) {
            if (window.showCustomMessage) {
                window.showCustomMessage('Tipo de Archivo Inválido', 'Por favor, sube un archivo WAV, MP3, M4A, OGG o WebM.', 'error');
            }
            this.resetUploadArea();
            return;
        }

        // Mostrar un mensaje de carga mientras se convierte
        if (window.showCustomMessage) {
            window.showCustomMessage('Cargando Audio', 'Procesando archivo de audio para compatibilidad...', 'info', 0); // Duración 0 = persistente
        }

        try {
            // Convertir el archivo a WAV antes de almacenarlo como audioBlob
            const wavBlob = await this.convertToWav(file);
            if (wavBlob) {
                this.audioBlob = wavBlob;
                // Usar un nombre de archivo .wav para el Blob convertido
                this.audioBlob.name = file.name.split('.').slice(0, -1).join('.') + '.wav'; 
                this.audioUrl = URL.createObjectURL(this.audioBlob);
                console.log("Archivo de audio seleccionado y convertido a WAV:", this.audioBlob.name);

                // Actualiza la UI para mostrar la información del archivo subido
                if (this.uploadedFileInfoDiv && this.fileNameSpan && this.fileSizeSpan && this.uploadZone) {
                    this.fileNameSpan.textContent = this.audioBlob.name; // Mostrar el nombre .wav
                    this.fileSizeSpan.textContent = `${(this.audioBlob.size / (1024 * 1024)).toFixed(2)} MB`;
                    this.uploadZone.classList.add('hidden');
                    this.uploadedFileInfoDiv.classList.remove('hidden');
                }
                
                // Asigna la URL al reproductor de audio principal de la sección de resultados
                if (this.audioPlayerElement) {
                    this.audioPlayerElement.src = this.audioUrl;
                    this.audioPlayerElement.load();
                }

                this.updateProcessButtonState(true);
            } else {
                throw new Error("No se pudo convertir el archivo a WAV.");
            }
        } catch (error) {
            console.error("Error al procesar el archivo de audio subido:", error);
            if (window.showCustomMessage) {
                window.showCustomMessage('Error al Cargar Audio', `Hubo un problema con el archivo: ${error.message}. Por favor, inténtalo de nuevo con otro archivo.`, 'error');
            }
            this.resetUploadArea();
        } finally {
            if (window.hideCustomMessage) {
                window.hideCustomMessage(); // Ocultar el mensaje de carga
            }
        }
    }

    /**
     * Reinicia el área de subida de archivos a su estado inicial.
     * También revoca la URL del Blob del audio subido anteriormente.
     */
    resetUploadArea() {
        this.revokeAudioUrl();
        this.audioBlob = null;
        if (this.audioFileInput) this.audioFileInput.value = '';
        if (this.uploadZone) this.uploadZone.classList.remove('hidden');
        if (this.uploadedFileInfoDiv) this.uploadedFileInfoDiv.classList.add('hidden');

        if (this.uploadZone) this.uploadZone.classList.remove('drag-over');

        this.updateProcessButtonState(false);
    }

    /**
     * Convierte un Blob de audio (ej. webm/opus, mp3) a un Blob WAV PCM.
     * Esto es crucial para asegurar la compatibilidad con el backend Python.
     * @param {Blob} audioBlobToConvert - El Blob de audio a convertir.
     * @param {number} sampleRate - La frecuencia de muestreo deseada para el WAV (ej. 44100 o 16000).
     * @returns {Promise<Blob|null>} Un Promise que resuelve a un Blob WAV o null si falla.
     */
    async convertToWav(audioBlobToConvert, sampleRate = 22050) { // Coincide con sr del backend
        if (!audioBlobToConvert) return null;

        // Crear un AudioContext temporal si no existe o ya está cerrado
        if (!this.audioContext || this.audioContext.state === 'closed') {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        try {
            const arrayBuffer = await audioBlobToConvert.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

            // Resamplear si la frecuencia de muestreo de audioBuffer es diferente a la deseada
            let resampledBuffer = audioBuffer;
            if (audioBuffer.sampleRate !== sampleRate) {
                const numberOfChannels = audioBuffer.numberOfChannels;
                const oldSampleRate = audioBuffer.sampleRate;
                const newSampleRate = sampleRate;
                const length = audioBuffer.length * newSampleRate / oldSampleRate;
                
                const offlineContext = new OfflineAudioContext(numberOfChannels, length, newSampleRate);
                const source = offlineContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(offlineContext.destination);
                source.start(0);
                resampledBuffer = await offlineContext.startRendering();
                console.log(`Audio re-sampleado de ${oldSampleRate}Hz a ${newSampleRate}Hz`);
            }

            // Convertir AudioBuffer a un Blob WAV
            const numChannels = resampledBuffer.numberOfChannels;
            const samples = resampledBuffer.getChannelData(0); // Asumimos mono, o tomamos el primer canal

            // Si es estéreo, podríamos promediar o simplemente tomar el primer canal para el modelo mono
            if (numChannels > 1) {
                console.warn("Se detectó audio estéreo. Se procesará solo el primer canal para la conversión a WAV mono.");
                // Opcional: Mezclar canales para una verdadera conversión mono
                // const mixedSamples = new Float32Array(resampledBuffer.length);
                // for (let i = 0; i < resampledBuffer.length; i++) {
                //     let sum = 0;
                //     for (let channel = 0; channel < numChannels; channel++) {
                //         sum += resampledBuffer.getChannelData(channel)[i];
                //     }
                //     mixedSamples[i] = sum / numChannels;
                // }
                // samples = mixedSamples;
            }

            const wavBuffer = this.encodeWAV(samples, resampledBuffer.sampleRate, numChannels); // Pasa numChannels
            return new Blob([wavBuffer], { type: 'audio/wav' });

        } catch (error) {
            console.error("Error al convertir audio a WAV:", error);
            if (window.showCustomMessage) {
                window.showCustomMessage('Error de Conversión', `No se pudo procesar el audio: ${error.message}. Intenta con otro archivo.`, 'error');
            }
            return null;
        } finally {
            // Asegurarse de cerrar el AudioContext temporal si fue creado para esta conversión
            if (this.audioContext && this.audioContext.state !== 'closed') {
                // Si el audioContext se usa para el visualizador, no lo cerramos aquí.
                // Lo cerramos solo si se creó específicamente para esta conversión y no está en uso continuo.
                // Para simplificar, lo dejaremos abierto para posibles futuras conversiones, y se cierra en resetRecordArea.
            }
        }
    }

    /**
     * Codifica un Array de Float32 a un ArrayBuffer WAV.
     * Función auxiliar para convertToWav.
     * @param {Float32Array} samples - Datos de audio.
     * @param {number} sampleRate - Frecuencia de muestreo.
     * @param {number} numChannels - Número de canales (1 para mono, 2 para estéreo).
     * @returns {ArrayBuffer} - El buffer del archivo WAV.
     */
    encodeWAV(samples, sampleRate, numChannels) {
        const buffer = new ArrayBuffer(44 + samples.length * 2); // 44 bytes para el encabezado WAV, 2 bytes por muestra (16-bit)
        const view = new DataView(buffer);

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        function floatTo16BitPCM(output, offset, input) {
            for (let i = 0; i < input.length; i++, offset += 2) {
                const s = Math.max(-1, Math.min(1, input[i]));
                output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
        }

        writeString(view, 0, 'RIFF'); // RIFF identifier
        view.setUint32(4, 36 + samples.length * 2, true); // file length (bytes) - 8
        writeString(view, 8, 'WAVE'); // RIFF type
        writeString(view, 12, 'fmt '); // format chunk identifier
        view.setUint32(16, 16, true); // format chunk length
        view.setUint16(20, 1, true); // sample format (1 = PCM)
        view.setUint16(22, numChannels, true); // number of channels
        view.setUint32(24, sampleRate, true); // sample rate
        view.setUint32(28, sampleRate * numChannels * 2, true); // byte rate (sample rate * block align)
        view.setUint16(32, numChannels * 2, true); // block align (num channels * bytes per sample)
        view.setUint16(34, 16, true); // bits per sample (16 bit)
        writeString(view, 36, 'data'); // data chunk identifier
        view.setUint32(40, samples.length * 2, true); // data chunk length

        floatTo16BitPCM(view, 44, samples);

        return buffer;
    }

    /**
     * Procesa el audio grabado o subido.
     * Esta función interactúa con el backend o un modelo de IA.
     */
    async processAudio() {
        if (!this.audioBlob) {
            if (window.showCustomMessage) {
                window.showCustomMessage('Error', 'No hay audio para procesar.', 'error');
            }
            return;
        }

        console.log("Procesando audio...");
        if (window.showLoadingSection) {
            window.showLoadingSection();
        }

        let audioToSend = this.audioBlob; // Por defecto, el Blob ya convertido a WAV
        let filename = this.audioBlob.name || 'audio.wav'; // Aseguramos extensión .wav

        // Si por alguna razón audioBlob no es WAV (ej. fallback en startRecording), se podría re-convertir aquí si es necesario
        // Pero la lógica de startRecording y handleFileSelect ya debería asegurar que audioBlob sea WAV.
        // Si quieres ser paranoico, podrías hacer otra conversión aquí:
        // if (this.audioBlob.type !== 'audio/wav') {
        //     audioToSend = await this.convertToWav(this.audioBlob);
        //     if (!audioToSend) {
        //         throw new Error("Fallo final en la conversión a WAV antes de enviar.");
        //     }
        //     filename = filename.split('.').slice(0, -1).join('.') + '.wav';
        // }


        try {
            const formData = new FormData();
            formData.append('audio', audioToSend, filename); // Enviar el Blob WAV

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errorDetails = 'Error desconocido.';
                try {
                    const errorData = await response.json();
                    errorDetails = errorData.error || errorDetails;
                } catch (jsonError) {
                    errorDetails = `Error HTTP ${response.status}: ${response.statusText}`;
                }
                throw new Error(errorDetails);
            }

            const result = await response.json();

            if (window.updateResultsSection) {
                window.updateResultsSection(result, this.audioUrl); 
            }

            if (window.showResultsSection) {
                window.showResultsSection();
            }
            if (window.showConfetti) {
                window.showConfetti();
            }

        } catch (error) {
            console.error("Error al procesar el audio:", error);
            if (window.showCustomMessage) {
                window.showCustomMessage('Error de Procesamiento',
                                         `Hubo un problema al analizar el audio: ${error.message}. Por favor, inténtalo de nuevo.`,
                                         'error');
            }
            if (window.showMainSection) {
                window.showMainSection();
            }
        } finally {
            this.audioBlob = null;
            this.updateProcessButtonState(false);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.audioRecorder = new AudioRecorder();
});