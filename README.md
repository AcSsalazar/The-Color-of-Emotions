# The Color of Emotions: Speech Emotion Recognition (SER)

> **Análisis sobre el rendimiento de modelos de *Deep Learning* entrenados a partir de datos numéricos mediante tensores 3D, imágenes 2D, y su subsecuente refuerzo mediante *early fusion*.**

---

##  Resumen (*Abstract*)
Este proyecto explora la intersección entre el análisis de audio y las representaciones visuales mediante el análisis de sonido emocional proveniente de fuentes multimodales. Abarca conceptos fundamentales de procesamiento de señales, reconocimiento de patrones y extracción de color, integrando además una perspectiva artística de estas transformaciones visuales. 

El objetivo principal es crear una colección de datos limpia y técnicamente precisa, disponible en dos formatos:
- **Imágenes 2D:** Espectrogramas de Mel y MFCC en formato PNG (resolución 224x224).
- **Tensores 3D:** Basados en coeficientes MFCC y sus deltas de primer y segundo orden.

---

## Objetivos

1. **Análisis preliminar:** Evaluación y preprocesamiento de muestras de sonido emocional.
2. **Extracción de características:** Obtención de descriptores acústicos clave (MFCC, Espectrogramas de Mel).
3. **Creación de *datasets* multimodales:** Generación de representaciones visuales y matriciales orientadas al entrenamiento de modelos.
4. **Reconocimiento y abstracción:** Clasificación de patrones emocionales utilizando redes neuronales y *computer vision*.
5. **Mirada artística:** Mapeo de intensidades frecuenciales hacia espacios de color para generar paletas visuales representativas de cada emoción.

---

##  Herramientas y Tecnologías

### Procesamiento de Audio
- **Librosa:** Análisis de audio y generación de espectrogramas de Mel.
- **Soundfile / SciPy:** Lectura y escritura de archivos de audio.
- **NumPy:** Computación numérica y manipulación de tensores.

### Procesamiento de Imagen y Visualización
- **Matplotlib / Seaborn / Plotly:** Visualización de espectrogramas y análisis estadístico.
- **Pillow (PIL) / Scikit-image:** Manipulación y procesamiento avanzado de imágenes.
- **ColorThief:** Extracción de paletas de color.

### Machine Learning & Deep Learning
- **Scikit-learn:** Reconocimiento de patrones y algoritmos de *clustering*.
- **TensorFlow / PyTorch:** Construcción y entrenamiento de modelos de *Deep Learning* para clasificación.

---

## Metodología y Fases del Proyecto

### Fase 1: Procesamiento de Audio y Extracción
- Carga de muestras de sonido emocional.
- Extracción de coeficientes MFCC y generación de espectrogramas de Mel.
- Normalización y preprocesamiento de datos.

### Fase 2: Análisis de Patrones
- Análisis estadístico para identificar patrones frecuenciales.
- Agrupación (*clustering*) de firmas emocionales similares.
- Extracción de características distintivas por emoción.

### Fase 3: Extracción de Color y Visión Artística
- Mapeo de intensidades de frecuencia a espacios de color (RGB, HSV, LAB).
- Generación de paletas de color específicas (5-10 colores por emoción) y gradientes temporales.
- Diseño de fondos estéticos y exportación de metadatos (PNG, HEX, JSON).

---

##  Preguntas de Investigación

El desarrollo de este proyecto busca responder a las siguientes interrogantes:
1. ¿Existen patrones frecuenciales consistentes para emociones específicas en diferentes sonidos?
2. ¿Pueden las paletas de color extraídas del sonido evocar visualmente la misma emoción?
3. ¿Cuál es la correlación entre los rangos de frecuencia y la percepción de calidez/frialdad del color?
4. ¿Cómo afectan las diferencias culturales a la percepción del sonido emocional y su asociación cromática?

---

##  Aplicaciones Prácticas

- **Diseño UI/UX:** Esquemas de color para interfaces basados en la conciencia emocional.
- **Visualización Musical:** Representación gráfica de emociones en tiempo real.
- **Aplicaciones Terapéuticas:** Apoyo visual para terapias de reconocimiento de emociones.
- **Arte Generativo:** Instalaciones artísticas impulsadas por estímulos sonoros emocionales.

---

##  Estructura del Repositorio

*(Estructura de directorios pendiente de actualización a medida que avance el proyecto).*

---

##  Cómo Contribuir
Las pautas de contribución (envío de muestras de sonido, estándares de calidad para paletas y código) estarán disponibles próximamente.

##  Licencia
[Por definir]

##  Contacto
**Mantenimiento del proyecto:** acsalazar-19@hotmail.com 
**Fecha de creación:** 11 de marzo de 2026  
**Estado:** Borrador - Segunda fase
