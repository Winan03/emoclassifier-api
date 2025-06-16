// animations.js - Sistema de Animaciones y Efectos Visuales para EmoClassifier

class AnimationController {
    constructor() {
        // Bandera para asegurar que la inicialización se ejecute una sola vez
        this.isInitialized = false;
        // Almacena los Intersection Observers para una limpieza adecuada
        this.observers = [];
        // Referencia a los elementos parallax para fácil acceso
        this.parallaxElements = null;
        // Bandera para controlar si las animaciones están activas (accesibilidad/rendimiento)
        this.animationsEnabled = true;

        this.init();
    }

    /**
     * Inicializa todas las animaciones y efectos visuales de la aplicación.
     * Se ejecuta solo una vez al cargar el DOM.
     */
    init() {
        if (this.isInitialized) {
            console.warn("AnimationController ya inicializado. Evitando doble inicialización.");
            return;
        }

        console.log("Inicializando AnimationController...");

        // Establece listeners para el control global de animaciones
        this.setupGlobalAnimationControl();

        // Inicializa las diferentes categorías de animaciones
        this.setupBackgroundAnimations();
        this.initializeScrollAnimations();
        this.setupParticleEffects();
        this.initializeHoverEffects();
        this.setupLoadingAnimations(); // Solo prepara los métodos, no inicia animaciones directamente
        this.initializeModalAnimations();

        this.isInitialized = true;
        console.log("AnimationController inicializado.");
    }

    /**
     * Configura el control global para pausar/reanudar todas las animaciones CSS.
     * Esto es útil para la accesibilidad o para ahorrar recursos.
     */
    setupGlobalAnimationControl() {
        // Exporta una función global para que otras partes de la app puedan controlar las animaciones
        window.toggleAllAnimations = (enable) => {
            if (typeof enable === 'boolean') {
                this.animationsEnabled = enable;
            } else {
                this.animationsEnabled = !this.animationsEnabled; // Alternar si no se especifica
            }

            document.body.classList.toggle('animations-paused', !this.animationsEnabled);
            console.log(`Animaciones ${this.animationsEnabled ? 'habilitadas' : 'pausadas'}.`);

            // Re-observar elementos si las animaciones se reanudan, para que se animen si están en vista
            if (this.animationsEnabled) {
                this.observers.forEach(observer => {
                    // Desconectar y volver a observar para forzar una re-evaluación
                    observer.disconnect();
                    document.querySelectorAll(`
                        .step-card, .feature, .emotion-card, .spectrogram-img,
                        .audio-player-card, .hero-title, .hero-subtitle,
                        .section-title, .about-text, .status-item, .tech-item, .probability-item
                    `).forEach(el => {
                        // Asegurar que el estado inicial de opacidad 0 esté en CSS si no está animado
                        el.style.animation = 'none'; // Eliminar animación inline para que se reproduzca de nuevo
                        el.offsetHeight; // Forzar reflow
                        observer.observe(el);
                    });
                });
            } else {
                // Pausar animaciones en curso si no se manejan con `animation-play-state`
                document.querySelectorAll('*').forEach(el => {
                    const computedStyle = getComputedStyle(el);
                    if (computedStyle.animationPlayState === 'running') {
                        el.style.animationPlayState = 'paused';
                    }
                });
            }
        };

        // Inject CSS for pausing animations globally
        this.injectStyles(`
            .animations-paused *, .animations-paused *::before, .animations-paused *::after {
                animation-play-state: paused !important;
                transition: none !important;
            }
        `, 'global-animation-pause-styles');
    }


    // ===== ANIMACIONES DE FONDO =====
    /**
     * Configura las animaciones de fondo, como las ondas y las partículas flotantes.
     */
    setupBackgroundAnimations() {
        // Inyecta los keyframes de las animaciones de fondo si no están en el CSS principal
        this.injectStyles(`
            @keyframes waveFloat {
                0% { transform: translate(-100%, 0) scale(1); }
                50% { transform: translate(100%, 0) scale(1.1); }
                100% { transform: translate(-100%, 0) scale(1); }
            }
            @keyframes float {
                0%, 100% { transform: translateY(0) translateX(0); }
                25% { transform: translateY(-10px) translateX(10px); }
                50% { transform: translateY(0) translateX(0); }
                75% { transform: translateY(10px) translateX(-10px); }
            }
        `, 'background-animation-keyframes');

        this.createWaveAnimations();
        this.setupFloatingElements();
    }

    /**
     * Crea y aplica animaciones a los elementos de onda en el fondo.
     * Asume que los keyframes 'waveFloat' están definidos en `styles.css`.
     */
    createWaveAnimations() {
        // Se asume que no hay elementos '.wave' en el HTML inicial y se crean aquí o se añaden manualmente si son necesarios.
        // Si existen en HTML, este método solo aplicaría la animación.
        // Por ahora, no hay elementos .wave en el HTML proporcionado, por lo que este método no hará nada directamente
        // a menos que se añadan dinámicamente o se modifique el HTML.
    }

    /**
     * Genera y añade elementos de partículas flotantes al body.
     * Asume que el keyframe 'float' está definido en `styles.css`.
     */
    setupFloatingElements() {
        const particleContainer = document.createElement('div');
        particleContainer.className = 'floating-particles';
        particleContainer.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none; z-index: -1; /* Detrás de todo el contenido */
        `;

        // Crea un número de partículas aleatorias
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.style.cssText = `
                position: absolute;
                width: ${Math.random() * 6 + 2}px; /* Tamaño aleatorio */
                height: ${Math.random() * 6 + 2}px;
                background: radial-gradient(circle, rgba(138, 43, 226, 0.5), transparent); /* Usar secondary-color */
                border-radius: 50%;
                left: ${Math.random() * 100}%; /* Posición horizontal aleatoria */
                top: ${Math.random() * 100}%; /* Posición vertical aleatoria */
                animation: float ${Math.random() * 6 + 4}s ease-in-out infinite; /* Duración aleatoria */
                animation-delay: ${Math.random() * 2}s; /* Retraso aleatorio */
            `;
            particleContainer.appendChild(particle);
        }
        document.body.appendChild(particleContainer);
    }

    // ===== ANIMACIONES DE SCROLL =====
    /**
     * Inicializa las animaciones que se activan al hacer scroll.
     */
    initializeScrollAnimations() {
        // Inyecta los keyframes de animaciones de entrada si no están en el CSS principal
        this.injectStyles(`
            @keyframes fadeInUp {
                0% { opacity: 0; transform: translateY(20px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            @keyframes fadeInDown {
                0% { opacity: 0; transform: translateY(-20px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            @keyframes slideInFromLeft {
                0% { opacity: 0; transform: translateX(-50px); }
                100% { opacity: 1; transform: translateX(0); }
            }
            @keyframes slideInFromRight {
                0% { opacity: 0; transform: translateX(50px); }
                100% { opacity: 1; transform: translateX(0); }
            }
            @keyframes scaleIn {
                0% { opacity: 0; transform: scale(0.8); }
                100% { opacity: 1; transform: scale(1); }
            }
            @keyframes rotateIn {
                0% { opacity: 0; transform: rotate(-45deg) scale(0.8); }
                100% { opacity: 1; transform: rotate(0deg) scale(1); }
            }
            @keyframes bounceIn {
                0%, 20%, 40%, 60%, 80%, 100% { transition-timing-function: cubic-bezier(0.215, 0.610, 0.355, 1.000); }
                0% { opacity: 0; transform: scale3d(0.3, 0.3, 0.3); }
                20% { transform: scale3d(1.1, 1.1, 1.1); }
                40% { transform: scale3d(0.9, 0.9, 0.9); }
                60% { opacity: 1; transform: scale3d(1.03, 1.03, 1.03); }
                80% { transform: scale3d(0.97, 0.97, 0.97); }
                100% { opacity: 1; transform: scale3d(1, 1, 1); }
            }
        `, 'scroll-animation-keyframes');

        this.setupIntersectionObserver();
        this.setupScrollProgress();
        this.setupParallaxEffect();
    }

    /**
     * Configura el Intersection Observer para animar elementos cuando entran en el viewport.
     * Aplica animaciones como 'fadeInUp', 'scaleIn', etc., definidas en CSS.
     */
    setupIntersectionObserver() {
        // Selecciona todos los elementos que tienen clases para ser animados al aparecer
        const animationTargets = document.querySelectorAll(`
            .step-card, .feature, .emotion-card, .spectrogram-img,
            .audio-player-card, .hero-title, .hero-subtitle,
            .section-title, .about-text, .status-item, .tech-item, .probability-item
        `);

        const observerOptions = {
            threshold: 0.1, // El 10% del elemento debe ser visible para activar
            rootMargin: '0px 0px -50px 0px' // Reduce el área del viewport para la detección
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting && this.animationsEnabled) {
                    // Aplica un retraso escalonado para un efecto visual agradable
                    const delay = index * 100;
                    setTimeout(() => {
                        this.animateElement(entry.target);
                        // Asegura que los elementos se vuelvan visibles después de la animación si estaban ocultos
                        entry.target.style.opacity = '1';
                    }, delay);
                    // Una vez animado, deja de observarlo para evitar re-animaciones
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        // Observa cada elemento objetivo
        animationTargets.forEach(target => {
            // Asegura que los elementos estén ocultos inicialmente para la animación
            target.style.opacity = '0';
            observer.observe(target);
        });

        // Almacena el observador para su posible limpieza
        this.observers.push(observer);
    }

    /**
     * Aplica una animación específica a un elemento, seleccionando el tipo de animación
     * basándose en las clases del elemento.
     * @param {HTMLElement} element - El elemento DOM a animar.
     */
    animateElement(element) {
        // Lista de animaciones CSS disponibles
        const animations = [
            'fadeInUp', 'fadeInDown', 'slideInFromLeft',
            'slideInFromRight', 'scaleIn', 'rotateIn', 'bounceIn'
        ];

        let animationType = 'fadeInUp'; // Animación por defecto

        // Lógica para elegir la animación según el tipo de elemento
        if (element.classList.contains('step-card') || element.classList.contains('tech-item')) {
            // Animación aleatoria para las tarjetas de pasos y tecnologías
            animationType = animations[Math.floor(Math.random() * animations.length)];
        } else if (element.classList.contains('hero-title')) {
            animationType = 'fadeInDown';
        } else if (element.classList.contains('hero-subtitle')) {
            animationType = 'fadeInUp';
        } else if (element.classList.contains('emotion-result') ||
                   element.classList.contains('spectrogram-section') ||
                   element.classList.contains('audio-section') ||
                   element.classList.contains('detailed-analysis') ||
                   element.classList.contains('action-buttons')) {
            animationType = 'scaleIn';
        } else if (element.classList.contains('status-item')) {
            animationType = 'bounceIn';
        } else if (element.classList.contains('section-title') ||
                   element.classList.contains('about-text')) {
            animationType = 'fadeInUp';
        } else if (element.classList.contains('probability-item')) {
            animationType = 'slideInFromLeft'; // Animación específica para barras de probabilidad
        }

        // Aplica la animación. 'forwards' mantiene el estado final de la animación (opacidad a 1).
        element.style.animation = `${animationType} 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards`;
    }

    /**
     * Configura una barra de progreso que indica el avance del scroll en la página.
     */
    setupScrollProgress() {
        const progressBar = document.createElement('div');
        progressBar.className = 'scroll-progress';
        // Inyecta los estilos para la barra de progreso
        this.injectStyles(`
            .scroll-progress {
                position: fixed;
                top: 0;
                left: 0;
                height: 4px;
                background: linear-gradient(90deg, var(--accent-color), var(--secondary-color));
                width: 0%;
                z-index: 9999; /* Por encima de todo, pero debajo del header fijo */
                transition: width 0.1s linear;
            }
        `, 'scroll-progress-styles');
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', () => {
            if (!this.animationsEnabled) return; // No actualizar si las animaciones están pausadas

            const scrolled = (window.scrollY /
                              (document.documentElement.scrollHeight - window.innerHeight)) * 100;
            progressBar.style.width = scrolled + '%';
        });
    }

    /**
     * Aplica un efecto parallax suave a los elementos de fondo hero y about al hacer scroll.
     */
    setupParallaxEffect() {
        this.parallaxElements = document.querySelectorAll('.hero-container'); // Apuntando al contenedor
        // Inyecta los estilos necesarios para las transformaciones parallax
        this.injectStyles(`
            .hero-container, .info-container {
                transition: transform 0.1s ease-out; /* Transición suave para el parallax */
            }
        `, 'parallax-styles');

        window.addEventListener('scroll', () => {
            if (!this.animationsEnabled) return; // No aplicar parallax si las animaciones están pausadas

            // Throttling para mejorar el rendimiento del scroll
            if (!this.scrollTimeout) {
                this.scrollTimeout = setTimeout(() => {
                    const scrolled = window.pageYOffset;
                    this.parallaxElements.forEach(element => {
                        // Ajusta la velocidad del efecto parallax
                        const rate = scrolled * -0.05; // Un valor más pequeño para un efecto sutil
                        element.style.transform = `translateY(${rate}px)`;
                    });
                    this.scrollTimeout = null;
                }, 10); // Límite a 10ms
            }
        });
    }

    // ===== EFECTOS DE HOVER =====
    /**
     * Inicializa los efectos de hover para diferentes tipos de elementos interactivos.
     */
    initializeHoverEffects() {
        this.setupButtonHovers();
        this.setupCardHovers();
        this.setupNavHovers(); // Menos JS, más CSS para estos
    }

    /**
     * Configura los efectos de hover y touch para botones.
     */
    setupButtonHovers() {
        const buttons = document.querySelectorAll(`
            .process-btn, .action-btn,
            .mode-btn, .record-btn, .control-btn, .remove-file-btn
        `);

        buttons.forEach(button => {
            // Los efectos de hover principales se manejan con CSS para mayor eficiencia.
            // JavaScript se usa para efectos 'active' o táctiles específicos.
            button.addEventListener('mousedown', () => {
                if (!this.animationsEnabled) return;
                button.style.transform = 'translateY(1px) scale(0.98)';
            });

            button.addEventListener('mouseup', () => {
                if (!this.animationsEnabled) return;
                button.style.transform = ''; // Vuelve al estado normal/hover CSS
            });
             button.addEventListener('mouseleave', () => {
                if (!this.animationsEnabled) return;
                button.style.transform = ''; // Asegura resetear si el mouse sale sin soltar
            });

            // Soporte táctil para efectos de 'presionado'
            button.addEventListener('touchstart', (e) => {
                if (!this.animationsEnabled) return;
                e.currentTarget.style.transform = 'translateY(1px) scale(0.98)';
            }, { passive: true });

            button.addEventListener('touchend', (e) => {
                if (!this.animationsEnabled) return;
                e.currentTarget.style.transform = '';
            }, { passive: true });
        });
    }

    /**
     * Configura los efectos de hover y touch para las tarjetas (cards).
     */
    setupCardHovers() {
        const cards = document.querySelectorAll(`
            .hero-container, .upload-zone, .uploaded-file-info,
            .step-card, .emotion-result, .spectrogram-section,
            .audio-section, .detailed-analysis, .tech-item
        `);

        cards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                if (!this.animationsEnabled) return;
                // Efecto de 'levantar' y 'rotar' ligeramente la tarjeta
                card.style.transform = 'translateY(-8px) rotateX(2deg)';
                card.style.boxShadow = '0 15px 35px rgba(0,0,0,0.4)'; // Sombra más pronunciada
            });

            card.addEventListener('mouseleave', () => {
                if (!this.animationsEnabled) return;
                card.style.transform = 'translateY(0) rotateX(0deg)';
                card.style.boxShadow = '0 10px 30px rgba(0,0,0,0.6)'; // Vuelve a la sombra original de .hero-container
            });

            // Soporte táctil
            card.addEventListener('touchstart', (e) => {
                if (!this.animationsEnabled) return;
                e.currentTarget.style.transform = 'translateY(-8px) rotateX(2deg)';
            }, { passive: true });

            card.addEventListener('touchend', (e) => {
                if (!this.animationsEnabled) return;
                e.currentTarget.style.transform = 'translateY(0) rotateX(0deg)';
            }, { passive: true });
        });
    }

    /**
     * Configura los efectos de hover para los enlaces de navegación.
     * La lógica principal del indicador de hover/activo debe estar en CSS (`::after`).
     */
    setupNavHovers() {
        // Los estilos del indicador de navegación son manejados por CSS (:hover::after, .active::after)
        // No se requiere lógica JS compleja aquí a menos que se desee un efecto muy específico no CSS.
    }

    // ===== EFECTOS DE PARTÍCULAS =====
    /**
     * Configura efectos visuales basados en partículas.
     */
    setupParticleEffects() {
        // Inyecta los keyframes de partículas si no están en el CSS principal
        this.injectStyles(`
            @keyframes particleExplosion {
                0% { transform: scale(1) translate(0, 0); opacity: 1; }
                100% { transform: scale(2) translate(var(--dx), var(--dy)); opacity: 0; }
            }
            @keyframes rippleEffect {
                0% { transform: translate(-50%, -50%) scale(0); opacity: 1; }
                100% { transform: translate(-50%, -50%) scale(2.5); opacity: 0; }
            }
        `, 'particle-keyframes');

        this.createClickParticles();
        this.setupRecordingParticles();
    }

    /**
     * Crea un efecto de explosión de partículas al hacer clic en cualquier lugar del documento.
     */
    createClickParticles() {
        document.addEventListener('click', (e) => {
            if (!this.animationsEnabled) return;
            // Evita el efecto de partículas en elementos específicos (ej. modales, el propio confeti)
            if (e.target.closest('.no-click-effect, .modal, .custom-message-modal, .confetti-piece')) {
                return;
            }

            const particle = document.createElement('div');
            // Calcula direcciones aleatorias para la explosión
            const angle = Math.random() * Math.PI * 2;
            const distance = Math.random() * 50 + 20; // Distancia entre 20 y 70px
            const dx = Math.cos(angle) * distance;
            const dy = Math.sin(angle) * distance;

            particle.style.cssText = `
                position: fixed;
                width: ${Math.random() * 8 + 4}px; height: ${Math.random() * 8 + 4}px;
                background: radial-gradient(circle, rgba(0, 188, 212, 0.8), transparent); /* Usar accent-color */
                border-radius: 50%;
                pointer-events: none; /* No debe interferir con la interacción del usuario */
                z-index: 10000;
                left: ${e.clientX - 4}px; /* Centra la partícula en el clic */
                top: ${e.clientY - 4}px;
                --dx: ${dx}px; /* Usar variables CSS para la animación */
                --dy: ${dy}px;
                animation: particleExplosion 0.6s ease-out forwards;
            `;

            document.body.appendChild(particle);

            // Elimina la partícula después de que la animación termine
            setTimeout(() => {
                if (particle.parentNode) {
                    particle.parentNode.removeChild(particle);
                }
            }, 600);
        });
    }

    /**
     * Configura el efecto de onda/ripple para el botón de grabación.
     */
    setupRecordingParticles() {
        const recordBtn = document.getElementById('recordBtn');

        if (recordBtn) {
            recordBtn.addEventListener('mousedown', () => {
                if (!this.animationsEnabled) return;
                this.createRecordingRipple(recordBtn);
            });
            recordBtn.addEventListener('touchstart', (e) => {
                if (!this.animationsEnabled) return;
                e.preventDefault(); // Previene el comportamiento por defecto del touch
                this.createRecordingRipple(e.currentTarget);
            }, { passive: false });
        }
    }

    /**
     * Crea un efecto de onda (ripple) en un elemento dado.
     * @param {HTMLElement} element - El elemento donde se creará el ripple.
     */
    createRecordingRipple(element) {
        const ripple = document.createElement('div');
        ripple.style.cssText = `
            position: absolute;
            width: 90%; height: 90%; /* Ripple cubre el 90% del botón */
            background: rgba(231, 76, 60, 0.3); /* Un poco de rojo para el efecto de grabación */
            border-radius: 50%;
            top: 50%; left: 50%; /* Centra el ripple en el elemento padre */
            transform: translate(-50%, -50%) scale(0);
            animation: rippleEffect 1s ease-out forwards;
            pointer-events: none;
            opacity: 0; /* Inicia invisible */
        `;

        // Asegura que el elemento padre tenga 'position: relative' o 'absolute'
        element.style.position = 'relative';
        element.style.overflow = 'hidden'; // Esconde el ripple fuera del botón
        element.appendChild(ripple);

        // Elimina el ripple después de la animación
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 1000);
    }

    // ===== ANIMACIONES DE CARGA =====
    /**
     * Prepara los métodos relacionados con las animaciones de carga.
     * La ejecución de estas animaciones es activada externamente (ej. por `main.js`).
     */
    setupLoadingAnimations() {
        // Inyecta los keyframes para los pasos de carga y confeti
        this.injectStyles(`
            @keyframes soundWaveAnim {
                0% { transform: scaleX(0); opacity: 0.2; }
                50% { transform: scaleX(1); opacity: 0.8; }
                100% { transform: scaleX(0); opacity: 0; }
            }
            @keyframes confettiFall {
                0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
                100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
            }
        `, 'loading-confetti-keyframes');

        // Expone la función para animar los pasos de carga
        window.animateLoadingSteps = () => this._animateLoadingSteps();
        // Expone la función para el confeti
        window.showConfetti = () => this.createConfetti();
    }

    /**
     * Anima los pasos individuales de la sección de carga.
     * Este método es llamado por EmoClassifier (a través de `window.animateLoadingSteps`).
     * Es un método interno (`_`) porque se expone globalmente.
     */
    _animateLoadingSteps() {
        const steps = document.querySelectorAll('.loading-steps .step-item'); // Asegúrate de usar la clase correcta
        const progressBarFill = document.querySelector('.progress-bar-container .progress-fill');
        const progressText = document.querySelector('.progress-bar-container .progress-text');

        if (!steps.length || !progressBarFill || !progressText || !this.animationsEnabled) {
            return;
        }

        // Reinicia el estado activo de todos los pasos
        steps.forEach(step => {
            step.classList.remove('active');
            step.style.opacity = '0.5'; // Iniciar un poco transparente
            step.style.transform = 'translateY(20px)'; // Pequeño desplazamiento inicial
            step.style.transition = 'none'; // Desactivar transición para reset rápido
        });
        progressBarFill.style.width = '0%';
        progressText.textContent = '0%';

        let currentStepIndex = 0;
        let progress = 0;
        const totalSteps = steps.length;
        const stepIntervalDuration = 1500; // Intervalo entre la activación de cada paso
        const progressPerStep = 100 / totalSteps;

        const interval = setInterval(() => {
            if (!this.animationsEnabled) { // Pausar si las animaciones están deshabilitadas
                return;
            }

            if (currentStepIndex < totalSteps) {
                const currentStep = steps[currentStepIndex];
                currentStep.classList.add('active');
                currentStep.style.opacity = '1';
                currentStep.style.transform = 'translateY(0)';
                currentStep.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out'; // Reactivar transición

                // Animar la barra de progreso
                progress += progressPerStep;
                progressBarFill.style.width = `${progress}%`;
                progressText.textContent = `${Math.round(progress)}%`;

                currentStepIndex++;
            } else {
                // Asegurar que la barra de progreso llegue al 100% al final
                progressBarFill.style.width = '100%';
                progressText.textContent = '100%';
                clearInterval(interval); // Detiene el intervalo cuando todos los pasos han sido animados
            }
        }, stepIntervalDuration);
    }

    // ===== ANIMACIONES DE MODAL =====
    /**
     * Inicializa las animaciones para mostrar y ocultar el modal.
     * Expone las funciones globales `showModalAnimated` y `hideModalAnimated`.
     */
    initializeModalAnimations() {
        const modal = document.getElementById('spectrogramModal');
        const modalContent = modal ? modal.querySelector('.modal-content') : null;
        const closeBtn = modal ? modal.querySelector('.modal-close') : null;

        if (modal && modalContent && closeBtn) {
            // Inyecta los estilos de animación para el modal
            this.injectStyles(`
                .modal {
                    transition: opacity 0.3s ease, visibility 0.3s ease;
                }
                .modal-content {
                    transition: transform 0.3s ease, opacity 0.3s ease;
                    transform: scale(0.7) rotateX(45deg); /* Estado inicial para la entrada */
                    opacity: 0;
                }
                .modal.active .modal-content {
                    transform: scale(1) rotateX(0deg);
                    opacity: 1;
                }
            `, 'modal-animation-styles');

            // Expone la función para mostrar el modal con animación
            window.showModalAnimated = () => {
                if (!this.animationsEnabled) {
                    modal.classList.add('active'); // Fallback sin animación si están desactivadas
                    modalContent.style.transform = 'scale(1) rotateX(0deg)';
                    modalContent.style.opacity = '1';
                    return;
                }

                modal.classList.add('active'); // Hace el modal visible (display: flex) y transiciona la opacidad
                // Pequeño retraso para que el navegador procese el 'display' antes de la transición
                setTimeout(() => {
                    modalContent.style.transform = 'scale(1) rotateX(0deg)';
                    modalContent.style.opacity = '1';
                }, 10);
            };

            // Expone la función para ocultar el modal con animación
            window.hideModalAnimated = () => {
                if (!this.animationsEnabled) {
                    modal.classList.remove('active'); // Fallback sin animación
                    return;
                }

                modalContent.style.transform = 'scale(0.7) rotateX(45deg)'; // Aplica la transformación de salida
                modalContent.style.opacity = '0';

                // Oculta el modal completamente después de que la animación de salida termine
                setTimeout(() => {
                    modal.classList.remove('active');
                }, 300); // Coincide con la duración de la transición en CSS
            };

            // Añade event listeners a los elementos para abrir/cerrar el modal
            closeBtn.addEventListener('click', () => {
                window.hideModalAnimated();
            });

            // Cierra el modal si se hace clic fuera del contenido del modal
            window.addEventListener('click', (e) => {
                if (e.target === modal) {
                    window.hideModalAnimated();
                }
            });
        } else {
            console.warn("Elementos del modal de espectrograma no encontrados. Las animaciones del modal no se inicializarán.");
        }
    }

    // ===== UTILIDADES Y MÉTODOS PÚBLICOS =====
    /**
     * Inyecta estilos CSS dinámicamente en el head del documento.
     * @param {string} cssText - El texto CSS a inyectar.
     * @param {string} id - Un ID para el elemento <style> para evitar duplicados.
     */
    injectStyles(cssText, id) {
        const existing = document.getElementById(id);
        if (existing) {
            existing.remove(); // Elimina el stylesheet existente con el mismo ID
        }

        const styleSheet = document.createElement('style');
        styleSheet.id = id;
        styleSheet.textContent = cssText;
        document.head.appendChild(styleSheet);
    }

    /**
     * Activa una animación CSS en un elemento dado.
     * Útil cuando necesitas re-disparar una animación o aplicar una dinámicamente.
     * @param {HTMLElement} element - El elemento DOM a animar.
     * @param {string} animationType - El nombre del keyframe CSS a aplicar (ej. 'fadeInUp').
     */
    triggerAnimation(element, animationType) {
        if (!element || !animationType || !this.animationsEnabled) {
            return;
        }

        // Resetea la animación para que se reproduzca de nuevo si ya estaba activa
        element.style.animation = 'none';
        // Forzar un reflow del navegador para asegurar que la animación se reinicie
        element.offsetHeight;
        element.style.animation = `${animationType} 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards`;
    }

    /**
     * Aplica un efecto de 'brillo' animado a un elemento.
     * @param {HTMLElement} element - El elemento al que se le aplicará el brillo.
     */
    addGlowEffect(element) {
        if (element && this.animationsEnabled) {
            // Inyecta el keyframe 'glow' si no está presente
            this.injectStyles(`
                @keyframes glow {
                    0%, 100% { box-shadow: 0 0 10px var(--accent-color), 0 0 20px var(--accent-color); }
                    50% { box-shadow: 0 0 25px var(--accent-color), 0 0 40px var(--accent-color); }
                }
            `, 'glow-keyframes');
            element.style.animation = 'glow 2s ease-in-out infinite alternate'; // 'alternate' para que brille de ida y vuelta
        }
    }

    /**
     * Elimina el efecto de 'brillo' de un elemento.
     * @param {HTMLElement} element - El elemento del que se removerá el brillo.
     */
    removeGlowEffect(element) {
        if (element) {
            element.style.animation = ''; // Limpia la propiedad de animación
        }
    }

    /**
     * Crea un efecto de confeti que cae desde la parte superior de la pantalla.
     */
    createConfetti() {
        if (!this.animationsEnabled) return;

        for (let i = 0; i < 50; i++) { // Crea 50 piezas de confeti
            const confetti = document.createElement('div');
            confetti.classList.add('confetti-piece'); // Añade una clase para identificar y estilizar si es necesario
            confetti.style.cssText = `
                position: fixed;
                width: ${Math.random() * 10 + 5}px; /* Tamaño aleatorio */
                height: ${Math.random() * 10 + 5}px;
                background: hsl(${Math.random() * 360}, 70%, 60%); /* Color aleatorio */
                left: ${Math.random() * 100}vw; /* Posición horizontal aleatoria */
                top: -10px; /* Empieza fuera de la pantalla */
                z-index: 10000;
                pointer-events: none; /* No debe interferir con la interacción del usuario */
                animation: confettiFall ${Math.random() * 3 + 2}s linear forwards; /* Duración aleatoria */
                animation-delay: ${Math.random() * 0.5}s; /* Pequeño retraso aleatorio para escalonar */
            `;

            document.body.appendChild(confetti);

            // Elimina la pieza de confeti después de su animación
            setTimeout(() => {
                if (confetti.parentNode) {
                    confetti.parentNode.removeChild(confetti);
                }
            }, 5000); // Un poco más de tiempo que la duración máxima de la animación para asegurar eliminación
        }
    }

    /**
     * Método de limpieza para desconectar Intersection Observers y eliminar elementos dinámicos.
     * Útil si la aplicación necesita ser reiniciada o sus componentes desechados.
     */
    cleanup() {
        // Desconecta todos los observadores de intersección
        this.observers.forEach(observer => observer.disconnect());
        this.observers = []; // Limpia el array de observadores

        // Elimina los elementos DOM creados dinámicamente
        document.querySelectorAll('.floating-particles, .scroll-progress, .confetti-piece').forEach(el => {
            if (el.parentNode) {
                el.parentNode.removeChild(el);
            }
        });
        // Si hay elementos '.wave' dinámicos, también eliminarlos.
        // En tu HTML actual no hay, pero si se añaden programáticamente:
        // document.querySelectorAll('.wave').forEach(wave => {
        //     if (wave.parentNode) {
        //         wave.parentNode.removeChild(wave);
        //     }
        // });
        console.log("AnimationController limpiado.");
    }
}

// Inicializa el controlador de animaciones cuando el DOM esté completamente cargado.
document.addEventListener('DOMContentLoaded', () => {
    window.animationController = new AnimationController();
});

// Expone los métodos de animación más utilizados globalmente para fácil acceso
// por otras clases como EmoClassifier o AudioRecorder.

/**
 * Dispara una animación CSS en un elemento.
 * @param {HTMLElement} element - El elemento DOM.
 * @param {string} type - El nombre del keyframe de la animación (ej. 'fadeInUp').
 */
window.triggerElementAnimation = (element, type) => {
    if (window.animationController) {
        window.animationController.triggerAnimation(element, type);
    }
};

/**
 * Muestra una lluvia de confeti.
 */
window.showConfetti = () => {
    if (window.animationController) {
        window.animationController.createConfetti();
    }
};

/**
 * Añade un efecto de brillo animado a un elemento.
 * @param {HTMLElement} element - El elemento DOM.
 */
window.addElementGlow = (element) => {
    if (window.animationController) {
        window.animationController.addGlowEffect(element);
    }
};

/**
 * Elimina el efecto de brillo de un elemento.
 * @param {HTMLElement} element - El elemento DOM.
 */
window.removeElementGlow = (element) => {
    if (window.animationController) {
        window.animationController.removeGlowEffect(element);
    }
};

/**
 * Inicia la animación de los pasos de carga.
 * Se asume que esta función se llama cuando la sección de carga se hace visible.
 */
window.animateLoadingSteps = () => {
    if (window.animationController && typeof window.animationController._animateLoadingSteps === 'function') {
        window.animationController._animateLoadingSteps();
    } else {
        console.warn("animateLoadingSteps no está disponible o AnimationController no está inicializado.");
    }
};

/**
 * Activa o desactiva todas las animaciones de la aplicación.
 * @param {boolean} [enable] - True para activar, False para desactivar. Si se omite, alterna el estado.
 */
window.toggleAllAnimations = (enable) => {
    if (window.animationController && typeof window.animationController.toggleAllAnimations === 'function') {
        window.animationController.toggleAllAnimations(enable);
    } else {
        console.warn("toggleAllAnimations no está disponible o AnimationController no está inicializado.");
    }
};