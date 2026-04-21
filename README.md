# The Color of Emotions: Speech Emotion Recognition (SER)

> **Análisis e implementación de redes neuronales entrenadas a partir de datos multimodales (*early fusion*), numéricos (tensores 3D) e imágenes 2D.**

---

##  Resumen (*Abstract*)
En este proyecto se abordan las diferencias en el entrenamiento de redes neuronales alimentadas por datos provenientes de señales de audio. Estas señales son transformadas en dos formatos distintos: datos numéricos multidimensionales mediante PyTorch e imágenes bidimensionales generadas a partir de sus espectrogramas de Mel.

A través de los capítulos de este proyecto se exploran conceptos fundamentales del procesamiento de señales, el reconocimiento de patrones y la extracción de características, integrando, además, una perspectiva artística del sonido concebido como imagen.

El objetivo principal en este repositorio es crear una colección de datos limpia y técnicamente precisa, disponible en dos formatos:

- [**Imágenes 2D:**](https://drive.google.com/file/d/16sWxbncvc9iazebQPa6vCrHe6KZkgUQS/view?usp=sharing) Espectrogramas de Mel y MFCC en formato PNG (resolución 224x224).
- [**Tensores 3D:**](https://drive.google.com/drive/folders/1GIuh9wlE3dxeL5lUlKGAkK66VWKCJU9L?usp=sharing) Basados en coeficientes MFCC y sus deltas de primer y segundo orden.

---

## Capitulos y Temas Abordados

1. **Análisis preliminar:** Exploración de descriptores acústicos con Librosa, reproducción de muestras de audio.
2. **Extracción de características:** Obtención, visualización y explicación de descriptores acústicos clave (MFCC, Espectrogramas de Mel).
3. **Creación de *datasets* multimodales:** Generación de representaciones visuales y matriciales orientadas al entrenamiento de modelos.
4. **Entrenamiento:** Entrenamiento de redes neuronales mediante los dos tipos de datos disponibles *computer vision* vs *audio processing*.
5. **Entrenamiento multimodal:** Evaluación de la eficiencia y precisión de los modelos entrenados con refuerzo (ResNet18 vs Effientnet-B0).
6. **Análisis de resultados:** Evaluacion del modelo con mejores resultados y visualización de patrones emocionales.
7. **Mirada artística:** Mapeo de intensidades frecuenciales hacia espacios de color para generar paletas visuales representativas de cada emoción.

---

##  Herramientas y Tecnologías

### Procesamiento de Audio
- **Librosa:** Análisis de audio y generación de espectrogramas de Mel.
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
- Data augmentation (ruido, cambio de velocidad) y combinacion de *datasets* para aumentar la diversidad del dataset final.
- Normalización y preprocesamiento de datos.

### Fase 2: Division de Datos y Análisis Exploratorio Final
- Análisis estadístico para identificar patrones frecuenciales.
- Agrupación (*clustering*) de clases entre los datos.
- Extracción de características clave por emoción.

### Fase 3: Extracción de Color y Visión Artística
- Mapeo de intensidades de frecuencia a espacios de color (RGB, HSV, LAB).
- Generación de paletas de color específicas (5-10 colores por emoción) y gradientes temporales.
- Diseño de fondos estéticos y exportación de metadatos (PNG, HEX, JSON).

---

##  Preguntas de Investigación

El desarrollo de este proyecto busca responder a las siguientes interrogantes:
1. ¿Existen patrones frecuenciales consistentes para emociones específicas en diferentes sonidos?
2. ¿Que tan eficiente resulta el entrenamiento de modelos de *Deep Learning* utilizando datos numéricos (tensores 3D) frente a imágenes (espectrogramas)?
3. ¿Hay marjen de mejora en la claisficacion tras implementar un método de refuerzo mediante datos multimodales con *early fusion*?
4. ¿Es posible generar paletas de color que representen visualmente las emociones a partir de sus características frecuenciales?
5. ¿Pueden estas paletas de color ser utilizadas en aplicaciones prácticas como diseño UI/UX, visualización musical o arte generativo?

---

##  Aplicaciones Prácticas

- **Diseño UI/UX:** Esquemas de color para interfaces basados en la conciencia emocional.
- **Detección en tiempo real:** Identificación de emociones en tiempo real para detectar cambios de humor.
- **Arte Generativo:** Esquema de ambientacion espacial basado en las paletas de color obtenidas.

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
