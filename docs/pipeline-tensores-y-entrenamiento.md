# Pipeline de Extracción de Tensores y Recomendaciones de Entrenamiento

**Proyecto:** The Color of Emotions  
**Referencia:** Notebook `3.2-feature-extraction-with-pytorch.ipynb`  
**Fecha de generación:** 2026-04-27  

---

## Índice

1. [Introducción](#1-introducción)  
2. [Orden del Pipeline de Preprocesamiento](#2-orden-del-pipeline-de-preprocesamiento)  
   2.1 [Resample](#21-resample)  
   2.2 [Trim por energía (−40 dB equivalente)](#22-trim-por-energía--40-db-equivalente)  
   2.3 [Normalización de amplitud](#23-normalización-de-amplitud)  
   2.4 [Descarte por duración mínima](#24-descarte-por-duración-mínima)  
   2.5 [Padding / Truncado a muestras fijas](#25-padding--truncado-a-muestras-fijas)  
   2.6 [Extracción de features (Mel dB / Delta / Cochlea)](#26-extracción-de-features-mel-db--delta--cochlea)  
   2.7 [Z-score por muestra y por canal](#27-z-score-por-muestra-y-por-canal)  
   2.8 [fix_frames: padding / truncado en frames](#28-fix_frames-padding--truncado-en-frames)  
3. [Trim y Normalización de Amplitud vs. Z-score: ¿por qué no son redundantes?](#3-trim-y-normalización-de-amplitud-vs-z-score-por-qué-no-son-redundantes)  
4. [Efectos del Z-score en Visualización con Matplotlib](#4-efectos-del-z-score-en-visualización-con-matplotlib)  
5. [Recomendaciones Prácticas de Entrenamiento](#5-recomendaciones-prácticas-de-entrenamiento)  
   5.1 [Carga de los archivos `.pt`](#51-carga-de-los-archivos-pt)  
   5.2 [Dataset y DataLoader](#52-dataset-y-dataloader)  
   5.3 [Arquitecturas CNN recomendadas](#53-arquitecturas-cnn-recomendadas)  
   5.4 [Regularización](#54-regularización)  
   5.5 [Manejo de desbalance e imbalance (clase `surprised`)](#55-manejo-de-desbalance-e-imbalance-clase-surprised)  
   5.6 [Métricas de evaluación](#56-métricas-de-evaluación)  
   5.7 [Split speaker-independent](#57-split-speaker-independent)  
   5.8 [Normalización adicional y leakage](#58-normalización-adicional-y-leakage)  
   5.9 [Sugerencias de hiperparámetros iniciales](#59-sugerencias-de-hiperparámetros-iniciales)  
6. [Cómo imprimir o exportar este documento](#6-cómo-imprimir-o-exportar-este-documento)  

---

## 1. Introducción

Este documento consolida la explicación del pipeline de extracción de características acústicas implementado en el notebook `3.2-feature-extraction-with-pytorch.ipynb` y añade recomendaciones prácticas para el entrenamiento de modelos con los tensores generados.

El pipeline utiliza **PyTorch**, **Torchaudio** y **nnAudio** (sin Librosa) para extraer un tensor de tres canales por cada archivo de audio, aprovechando la aceleración por GPU. Los datasets empleados son **RAVDESS** y **CREMA-D**, con un total aproximado de **8 882 archivos `.wav`**.

### Parámetros globales de referencia

| Parámetro         | Valor   | Descripción                                         |
|-------------------|---------|-----------------------------------------------------|
| `SAMPLE_RATE`     | 16 000  | Hz (limitado por CREMA-D)                           |
| `MIN_DURATION`    | 0.5 s   | Duración mínima tras trim; muestras menores se descartan |
| `MAX_DURATION`    | 1.6 s   | Duración máxima tras trim                           |
| `N_MELS`          | 60      | Bandas mel / bins gammatone                         |
| `HOP_LENGTH`      | 512     | Salto en muestras para STFT                         |
| `TARGET_SAMPLES`  | 25 600  | `SAMPLE_RATE × MAX_DURATION`                        |
| `TARGET_FRAMES`   | 51      | `TARGET_SAMPLES / HOP_LENGTH + 1`                   |

La elección de `MAX_DURATION = 1.6 s` se justifica mediante la relación:

```
MAX_DURATION = TARGET_FRAMES × HOP_LENGTH / SAMPLE_RATE
             = 50 × 512 / 16 000
             = 1.6 s
```

Una inspección visual del dataset reveló que en la gran mayoría de los archivos la información acústica relevante se concentra entre los frames 30 y 50. Ajustar la duración máxima a 1.6 s garantiza exactamente **50 frames útiles** (TARGET_FRAMES = 51 tras la fórmula con `+ 1`), eliminando silencios redundantes y optimizando la eficiencia de entrada al modelo.

---

## 2. Orden del Pipeline de Preprocesamiento

El pipeline se aplica de manera secuencial a cada archivo `.wav`. A continuación se detalla cada paso.

### 2.1 Resample

**¿Qué hace?** Convierte la tasa de muestreo original del archivo a `SAMPLE_RATE = 16 000 Hz`.

**¿Por qué?** CREMA-D opera a 16 kHz, mientras que RAVDESS puede variar. Homogeneizar la frecuencia de muestreo garantiza que todos los espectrogramas tengan la misma resolución temporal y espectral.

```python
if sr != SAMPLE_RATE:
    waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
```

---

### 2.2 Trim por energía (−40 dB equivalente)

**¿Qué hace?** Elimina los silencios al inicio y al final de la señal, conservando solo la región donde hay contenido acústico relevante.

**Implementación en PyTorch:**  
A diferencia de Librosa (que opera con `top_db`), aquí se computa un umbral dinámico basado en el valor pico de la onda:

```python
threshold = max(|waveform|) × 0.01   # equivalente a −40 dB
```

La relación logarítmica que justifica el factor `0.01`:

```
20 × log₁₀(0.01) = −40 dB
```

Solo se consideran muestras cuya amplitud supera este umbral. Adicionalmente, antes de intentar el trim se verifica que la energía cuadrática media (MSE) supere `1 × 10⁻⁶`, asegurando que la señal tenga contenido suficiente:

```python
mse = torch.mean(waveform ** 2)
if mse > 1e-6:
    threshold = torch.max(torch.abs(waveform)) * 0.01
    mask = torch.abs(waveform[0]) > threshold
    idx  = torch.where(mask)[0]
    if len(idx) > 0:
        waveform = waveform[:, idx[0]:idx[-1] + 1]
```

**¿Por qué es importante?** Los silencios al inicio y al final no aportan información emocional. Su eliminación reduce el ruido de fondo, mejora la relación señal-ruido del espectrograma resultante y permite concentrar los frames útiles en el rango `MAX_DURATION`.

---

### 2.3 Normalización de amplitud

**¿Qué hace?** Escala la forma de onda al rango **[−1, 1]** dividiendo por su valor pico absoluto:

```python
peak = torch.max(torch.abs(waveform))
if peak > 1e-8:
    waveform = waveform / peak
```

**¿Por qué?** Garantiza que todos los archivos de audio tengan el mismo nivel de potencia pico antes de extraer features. Sin esta normalización, diferencias de volumen entre grabaciones provocarían escalas muy distintas en los espectrogramas, dificultando el aprendizaje.

> **Nota:** esta operación actúa sobre la señal en el **dominio temporal** (muestras). Es una normalización de **amplitud**, distinta al z-score que se aplica posteriormente sobre las features en el dominio espectral. Ver la Sección 3 para la diferencia completa.

---

### 2.4 Descarte por duración mínima

**¿Qué hace?** Descarta cualquier archivo cuya duración —tras el trim— sea inferior a `MIN_DURATION = 0.5 s`:

```python
min_samples = int(MIN_DURATION * SAMPLE_RATE)   # 8 000 muestras
if waveform.shape[1] < min_samples:
    return None, False
```

**¿Por qué?** Audios muy cortos (menos de 0.5 s) no contienen suficiente información temporal para construir un espectrograma representativo con los parámetros definidos, y podrían introducir tensores degenerados.

---

### 2.5 Padding / Truncado a muestras fijas

**¿Qué hace?** Ajusta la longitud de la forma de onda a exactamente `TARGET_SAMPLES = 25 600` muestras:

- Si la señal es **más corta**: se añaden ceros de manera centrada (*padding* simétrico).
- Si la señal es **más larga**: se trunca por el extremo derecho.

```python
n = waveform.shape[1]
if n > TARGET_SAMPLES:
    waveform = waveform[:, :TARGET_SAMPLES]
elif n < TARGET_SAMPLES:
    pad = TARGET_SAMPLES - n
    waveform = F.pad(waveform, (pad // 2, pad - pad // 2), mode='constant')
```

**¿Por qué centrado?** El padding centrado distribuye los ceros de manera simétrica, evitando que la voz quede pegada a un extremo del espectrograma, lo que mejora la representación espacial.

---

### 2.6 Extracción de features (Mel dB / Delta / Cochlea)

**¿Qué hace?** Aplica la clase `AudioFeatureExtractor` para producir un tensor de forma **[3, N_MELS, frames]** con tres canales:

| Canal | Descripción                                                |
|-------|------------------------------------------------------------|
| 0     | **Mel-Spectrogram en dB** — captura la distribución espectral de energía en escala perceptual. |
| 1     | **Delta temporal del Mel** — derivada de primer orden del Mel; captura la dinámica temporal (velocidad de cambio). |
| 2     | **Cochleagrama (Gammatonegram) en dB** — representación basada en la percepción auditiva humana. |

```python
class AudioFeatureExtractor(torch.nn.Module):
    def __init__(self, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sr, n_fft=512, hop_length=hop_length,
            n_mels=n_mels, f_min=20
        )
        self.cochlear = Spectrogram.Gammatonegram(
            sr=sr, n_fft=512, n_bins=n_mels, fmin=20
        )
        self.to_db = T.AmplitudeToDB()

    def forward(self, x):
        """x: [1, T]  →  [3, N_MELS, frames]"""
        mel     = self.to_db(self.mel_spec(x))      # [1, N_MELS, F]
        delta   = torchaudio.functional.compute_deltas(mel)
        cochlea = self.to_db(self.cochlear(x))       # [1, N_MELS, F]
        out = torch.cat([mel, delta, cochlea], dim=0)  # [3, N_MELS, F]
        return out
```

**¿Por qué estos tres canales?**
- El **Mel-Spectrogram** demostró buen rendimiento en pruebas previas con archivos PNG.  
- La **Delta** añade información de movimiento espectral, útil para distinguir emociones con patrones de variación temporal distintos.  
- El **Cochleagrama** incorpora una perspectiva perceptual diferente (banco de filtros Gammatone) que complementa al Mel, dotando al tensor de mayor riqueza informativa.

Esta combinación replica, de manera aproximada, la variedad de canales que se usaba antes con MFCC + Delta + Delta-Delta, pero con representaciones espectrales más ricas.

---

### 2.7 Z-score por muestra y por canal

**¿Qué hace?** Normaliza estadísticamente cada canal del tensor de manera independiente:

```python
def zscore_per_channel(tensor):
    eps = 1e-8
    for c in range(tensor.shape[0]):   # c = 0, 1, 2
        mean = tensor[c].mean()
        std  = tensor[c].std()
        tensor[c] = (tensor[c] - mean) / (std + eps)
    return tensor
```

Resultado: cada canal tiene **media ≈ 0** y **desviación estándar ≈ 1**.

**¿Por qué?** Aunque el Mel y el Cochlea están en dB y la Delta es adimensional, sus rangos numéricos difieren considerablemente entre muestras (por características de la grabación) y entre canales. El z-score estandariza estos rangos para facilitar el aprendizaje de la red. Ver la Sección 3 para la diferencia con la normalización de amplitud.

---

### 2.8 fix_frames: padding / truncado en frames

**¿Qué hace?** Garantiza que el tensor tenga exactamente `TARGET_FRAMES` frames en la dimensión temporal, independientemente de pequeñas variaciones producidas por el STFT:

```python
def fix_frames(tensor, target_frames=TARGET_FRAMES):
    f = tensor.shape[2]
    if f < target_frames:
        tensor = F.pad(tensor, (0, target_frames - f), mode='constant', value=0.0)
    elif f > target_frames:
        tensor = tensor[:, :, :target_frames]
    return tensor
```

Tras este paso, cada tensor tiene forma exacta **[3, 60, 51]**.  
Se valida con `assert feat.shape == (3, N_MELS, TARGET_FRAMES)` para prevenir el bug de canales duplicados (`[6, N_MELS, F]`) que ocurría en implementaciones anteriores.

---

## 3. Trim y Normalización de Amplitud vs. Z-score: ¿por qué no son redundantes?

Aunque ambas operaciones "normalizan" algo, actúan sobre **dominios distintos** y con **objetivos distintos**:

| Aspecto                  | Trim + Normalización de Amplitud                               | Z-score por canal                                                |
|--------------------------|----------------------------------------------------------------|------------------------------------------------------------------|
| **Dominio**              | Señal temporal (muestras de audio)                             | Features espectrales (espectrograma, tras STFT)                  |
| **Qué se escala**        | La amplitud cruda del audio (valores entre −1 y 1)             | Los valores de energía / dB de cada canal del tensor             |
| **Objetivo principal**   | Homogeneizar el volumen entre grabaciones; eliminar silencios  | Centrar y escalar la distribución de features para la red        |
| **Estadísticos usados**  | Valor pico absoluto de la forma de onda                        | Media y desviación estándar de cada canal por muestra            |
| **Riesgo de leakage**    | No aplica (es por muestra)                                     | Ninguno (es por muestra y por canal, no requiere stats de train) |
| **Resultado**            | Señal en [−1, 1]; duración homogénea                          | Canal con media ≈ 0 y std ≈ 1                                    |

### ¿Por qué ambas son necesarias?

1. **Sin trim + normalización de amplitud:** dos grabaciones del mismo fonema con volúmenes muy distintos producirían espectrogramas a escalas dB totalmente diferentes, incluso tras el z-score (porque el z-score centra y escala, pero si la señal tiene 20 dB más de potencia base, los valores de dB siguen siendo comparativamente grandes antes del z-score).

2. **Sin z-score:** aunque los audios estén normalizados en amplitud, los tres canales (Mel dB, Delta, Cochlea dB) tienen rangos numéricos muy distintos entre sí. El z-score garantiza que los gradientes fluyan de manera equilibrada a través de todos los canales durante el entrenamiento.

3. **¿Por qué z-score por muestra y no por dataset?** Si se calculara la media y desviación estándar globales del set de entrenamiento (estilo `StandardScaler`), habría **data leakage** potencial (estadísticos calculados sobre todo el train se aplican a val/test, introduciendo información implícita). El z-score por muestra y por canal es **completamente independiente de otros ejemplos**, eliminando el problema de leakage y siendo igualmente efectivo en la práctica.

---

## 4. Efectos del Z-score en Visualización con Matplotlib

Tras el z-score, los valores de cada canal ya **no están en dB** ni en un rango fijo predecible; se distribuyen aproximadamente alrededor de 0 con desviación ≈ 1. Esto tiene consecuencias para la visualización.

### Recomendación de `vmin` / `vmax`

```python
import matplotlib.pyplot as plt

# Ejemplo: visualizar canal 0 (Mel dB tras z-score)
tensor = ...  # shape [3, N_MELS, TARGET_FRAMES]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ['Mel dB (z-score)', 'Delta (z-score)', 'Cochlea dB (z-score)']

for i, ax in enumerate(axes):
    im = ax.imshow(
        tensor[i].numpy(),
        aspect='auto',
        origin='lower',
        cmap='magma',
        vmin=-3,   # ← 3 desviaciones estándar por debajo de la media
        vmax=3     # ← 3 desviaciones estándar por encima de la media
    )
    ax.set_title(titles[i])
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

**¿Por qué `vmin=-3, vmax=3`?**  
Bajo una distribución aproximadamente normal, el 99.7 % de los valores cae en el intervalo [−3σ, +3σ]. Fijar estos límites evita que valores atípicos (outliers) colapsen el contraste visual. Si se omite `vmin`/`vmax`, Matplotlib usará el mínimo y máximo absolutos, lo cual puede provocar imágenes muy planas donde los detalles no se aprecian.

**Valores alternativos según el caso:**
- `vmin=-2, vmax=2` para un contraste más agresivo (resalta variaciones pequeñas).  
- `vmin=-4, vmax=4` si hay muchos valores extremos (audios con ruido o grabaciones difíciles).

**Nota sobre la Delta (Canal 1):**  
La Delta puede tener una distribución más asimétrica o con colas más largas que el Mel o la Cochlea. Si la visualización se ve plana, prueba `vmin=-2, vmax=2` específicamente para ese canal.

---

## 5. Recomendaciones Prácticas de Entrenamiento

### 5.1 Carga de los archivos `.pt`

Los archivos generados por el pipeline tienen el siguiente formato:

```python
{
    "x":            torch.Tensor,   # shape [N, 3, N_MELS, TARGET_FRAMES] = [N, 3, 60, 51], float32
    "y":            torch.Tensor,   # shape [N], long (índices de clase)
    "meta":         list[dict],     # N diccionarios con: filename, actor_id, emotion_str,
                                    #   split, source, version
    "class_to_idx": dict,           # {'angry': 0, 'disgust': 1, 'fearful': 2, 'happy': 3,
                                    #   'neutral': 4, 'sad': 5, 'surprised': 6}
    "config":       dict            # sr, hop_length, max_duration, n_mels, target_frames, etc.
}
```

**Carga básica:**

```python
import torch

train_pack = torch.load("train_tensors.pt", map_location="cpu", weights_only=False)
val_pack   = torch.load("val_tensors.pt",   map_location="cpu", weights_only=False)
test_pack  = torch.load("test_tensors.pt",  map_location="cpu", weights_only=False)

X_train = train_pack["x"]           # [7422, 3, 60, 51]
y_train = train_pack["y"]           # [7422]
meta_train = train_pack["meta"]     # lista de dicts

class_to_idx = train_pack["class_to_idx"]
config       = train_pack["config"]

print("Clases:", class_to_idx)
print("Config:", config)
print("Shape X_train:", X_train.shape)
```

---

### 5.2 Dataset y DataLoader

#### Dataset mínimo

```python
import torch
from torch.utils.data import Dataset, DataLoader

class PackedTensorDataset(Dataset):
    def __init__(self, pack):
        self.x    = pack["x"]     # [N, 3, 60, 51]
        self.y    = pack["y"]     # [N]
        self.meta = pack["meta"]  # list[dict]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]  # (tensor [3, 60, 51], label int64)
```

#### DataLoader recomendado

```python
BATCH_SIZE  = 64
NUM_WORKERS = 4   # ajustar según núcleos disponibles en tu SSD local

train_ds = PackedTensorDataset(train_pack)
val_ds   = PackedTensorDataset(val_pack)
test_ds  = PackedTensorDataset(test_pack)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,               # aleatorizar en cada epoch (entrenamiento)
    num_workers=NUM_WORKERS,
    pin_memory=True,            # acelera transferencia CPU → GPU
    drop_last=True,             # evita batch incompleto al final
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,              # no aleatorizar en validación / test
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
```

#### Uso de GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    # ... forward pass
```

**Parámetros clave:**
- `shuffle=True` solo en `train_loader`; nunca en val/test.
- `pin_memory=True` es útil cuando el modelo está en GPU y los datos vienen de CPU.
- `num_workers` entre 2 y 8 es típico para SSD local; ajustar si hay cuello de botella.
- `drop_last=True` en train evita BatchNorm con batch de tamaño 1 al final de la época.

---

### 5.3 Arquitecturas CNN recomendadas

Los tensores tienen forma **[N, 3, 60, 51]**: 3 canales, 60 filas (frecuencia), 51 columnas (tiempo). Se tratan como imágenes de 3 canales, lo que permite usar arquitecturas CNN estándar.

#### A) Baseline simple (CNN desde cero)

```python
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # → [32, 30, 25]

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # → [64, 15, 12]

            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),           # → [128, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # → [2048]
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**Ventajas:** rápido de entrenar, fácil de interpretar, buen punto de partida.

#### B) ResNet18 / EfficientNet-B0 adaptado (transfer learning)

```python
import torchvision.models as models

# ResNet18 con primera capa adaptada a 3 canales y entrada [3, 60, 51]
def build_resnet18(num_classes=7, pretrained=True):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    # La primera capa ya acepta 3 canales → no se modifica
    # Adaptar el clasificador final
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

# EfficientNet-B0
def build_efficientnet(num_classes=7, pretrained=True):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model
```

**Nota sobre el tamaño de entrada:**  
ResNet y EfficientNet esperan entradas de `224×224` por defecto, pero son totalmente convolucionales hasta el `AdaptiveAvgPool2d`, por lo que aceptan entradas de cualquier tamaño ≥ un mínimo (generalmente 32×32). Con `[3, 60, 51]` funciona sin modificación adicional.

**Nota sobre preentrenamiento:**  
Los pesos de ImageNet fueron aprendidos con imágenes RGB. Los espectrogramas z-scoreados tienen una distribución diferente, pero el transfer learning sigue siendo beneficioso para aprender filtros de bajo nivel (bordes, texturas). Si el dataset es suficientemente grande, también es válido entrenar desde cero (`pretrained=False`).

---

### 5.4 Regularización

Para evitar overfitting y mejorar la generalización:

| Técnica          | Recomendación                                                                  |
|------------------|--------------------------------------------------------------------------------|
| **BatchNorm**    | Incluir después de cada capa convolucional; estabiliza el entrenamiento.       |
| **Dropout**      | 0.4–0.5 antes de la capa lineal final; 0.2–0.3 en capas intermedias.          |
| **Weight Decay** | `1e-4` a `1e-3` en el optimizador (L2 regularización).                        |
| **Early Stopping** | Detener si `val_loss` no mejora en 10–15 épocas consecutivas; restaurar el mejor checkpoint. |

**Ejemplo de optimizador con weight decay:**

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
```

---

### 5.5 Manejo de desbalance e imbalance (clase `surprised`)

La clase `surprised` está significativamente subrepresentada en el dataset (solo proveniente de RAVDESS, no de CREMA-D). El pipeline aplica **data augmentation fijo** en `train`:

- `original`: muestra base  
- `noise`: ruido AWGN leve  
- `shift`: desplazamiento circular ±5 000 muestras  

Esto triplica las muestras de `surprised` en train (432 vs 144 originales). **Val y test nunca contienen versiones sintéticas.**

**Técnicas adicionales recomendadas:**

1. **Class weights en la función de pérdida:**

```python
from torch.nn import CrossEntropyLoss
import torch

# Calcular pesos inversamente proporcionales a la frecuencia de clase
counts = torch.bincount(y_train)                    # [7]
weights = 1.0 / counts.float()
weights = weights / weights.sum()                   # normalizar
weights = weights.to(device)

criterion = CrossEntropyLoss(weight=weights)
```

2. **Weighted sampler** (si se prefiere sobre-muestreo dinámico):

```python
from torch.utils.data import WeightedRandomSampler

sample_weights = weights[y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
```

> **Nota:** No usar `shuffle=True` si se usa `sampler`; son mutuamente excluyentes en PyTorch.

---

### 5.6 Métricas de evaluación

Para un problema de clasificación multiclase desbalanceada, las métricas recomendadas son:

| Métrica                   | Justificación                                                        |
|---------------------------|----------------------------------------------------------------------|
| **Balanced Accuracy**     | Media del recall por clase (equivale a accuracy macro-promediado); penaliza por igual cada clase; robusto ante desbalance. Es `sklearn.metrics.balanced_accuracy_score`. |
| **Accuracy global**       | Porcentaje de aciertos totales; útil como referencia pero sesgado hacia clases mayoritarias. Es `sklearn.metrics.accuracy_score`. |
| **F1 macro**              | Media de F1 por clase; equilibra precisión y recall; robusto ante desbalance. |
| **Confusion Matrix**      | Permite identificar qué pares de emociones se confunden.             |

**Implementación con scikit-learn:**

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import numpy as np

# Inferencia
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        logits  = model(x_batch)
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

acc_global   = accuracy_score(all_labels, all_preds)            # accuracy global
acc_balanced = balanced_accuracy_score(all_labels, all_preds)   # media del recall por clase
f1           = f1_score(all_labels, all_preds, average='macro')
cm           = confusion_matrix(all_labels, all_preds)

print(f"Accuracy global:    {acc_global:.4f}")
print(f"Balanced Accuracy:  {acc_balanced:.4f}")   # la métrica preferida para conjuntos desbalanceados
print(f"F1 (macro):         {f1:.4f}")
print("Confusion Matrix:")
print(cm)
```

**Visualización de la Confusion Matrix:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

idx_to_class = {v: k for k, v in class_to_idx.items()}
labels = [idx_to_class[i] for i in range(len(class_to_idx))]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()
```

---

### 5.7 Split speaker-independent

El split realizado en el notebook 3.2 es **speaker-independent**: los actores asignados a `train`, `val` y `test` son **disjuntos**. Esto es crítico para evaluar la capacidad de **generalización** del modelo.

| Split | Actores | Muestras aprox. |
|-------|---------|-----------------|
| Train | 80 %    | 7 422           |
| Val   | 10 %    | 836             |
| Test  | 10 %    | 912             |

**¿Por qué importa?**  
Si un mismo actor aparece en train y test, el modelo puede "memorizar" características de voz específicas del actor (timbre, acento) en lugar de aprender a reconocer la emoción. El split por actor garantiza que las métricas de test reflejen rendimiento real en hablantes desconocidos.

**Implicación práctica:** Al reportar resultados, siempre indicar que el split es speaker-independent. Comparar con baselines que usan el mismo protocolo.

---

### 5.8 Normalización adicional y leakage

El z-score ya fue aplicado **por muestra y por canal** durante la generación de tensores. Esto significa:

- **No aplicar un segundo z-score** durante el entrenamiento (DataLoader / Dataset); sería redundante y podría introducir inestabilidades numéricas.
- **No computar estadísticos globales** (media/std de todo el train) y aplicarlos al val/test, ya que eso constituiría **data leakage** (información de val/test influye implícitamente en el preprocesamiento).
- Si en el futuro se desea un z-score global (estilo `StandardScaler`), los estadísticos deben calcularse **solo sobre train** y aplicarse a val y test sin refitting.

**Resumen de lo que ya está hecho y lo que no hace falta:**

| Operación                        | ¿Aplica? | Notas                                               |
|----------------------------------|----------|-----------------------------------------------------|
| Resample                         | ✅ Hecho  | Durante generación del tensor                       |
| Trim de silencios (−40 dB)       | ✅ Hecho  | Durante generación del tensor                       |
| Normalización de amplitud        | ✅ Hecho  | Durante generación del tensor                       |
| Z-score por muestra y por canal  | ✅ Hecho  | Durante generación del tensor; sin leakage          |
| Augmentation (surprised/train)   | ✅ Hecho  | Offline y fijo; val/test no tienen versiones sintéticas |
| Z-score global (StandardScaler)  | ❌ No aplicar | Podría introducir leakage; no es necesario    |
| Normalización adicional en DataLoader | ❌ No aplicar | El tensor ya está normalizado                |

---

### 5.9 Sugerencias de hiperparámetros iniciales

Los valores siguientes son un punto de partida razonable. Deben ajustarse mediante búsqueda o experimentación.

| Hiperparámetro          | Valor sugerido     | Rango a explorar         |
|-------------------------|--------------------|--------------------------|
| `batch_size`            | 64                 | 32, 64, 128              |
| `learning_rate`         | 1e-3               | 1e-4, 5e-4, 1e-3         |
| `weight_decay`          | 1e-4               | 1e-5, 1e-4, 1e-3         |
| `dropout`               | 0.4 (capa final)   | 0.2, 0.3, 0.4, 0.5       |
| `num_epochs`            | 50 (con early stopping) | —                   |
| `patience` (early stopping) | 10             | 5, 10, 15                |
| `lr_scheduler`          | ReduceLROnPlateau (factor=0.5, patience=5) | CosineAnnealingLR |
| `optimizer`             | Adam               | AdamW, SGD con momentum  |

**Flujo de experimentación recomendado:**

1. Entrenar el **Baseline CNN** durante 30–50 épocas con los hiperparámetros sugeridos.  
2. Comparar con **ResNet18 preentrenado** (fine-tuning de la última capa o de todo el modelo).  
3. Ajustar `learning_rate` y `weight_decay` con un grid search o búsqueda aleatoria sobre val.  
4. Elegir el modelo con mayor **F1 macro** en val; evaluarlo una sola vez en test.

---

## 6. Cómo imprimir o exportar este documento

Este archivo está en formato **Markdown** (`.md`), que se puede visualizar directamente en GitHub. Para imprimirlo o convertirlo a PDF:

### Opción A: Desde el navegador (GitHub)
1. Abre el archivo en GitHub (render automático de Markdown).  
2. `Ctrl + P` → Imprimir → selecciona "Guardar como PDF".

### Opción B: Con Pandoc (línea de comandos)
```bash
pandoc docs/pipeline-tensores-y-entrenamiento.md \
       -o docs/pipeline-tensores-y-entrenamiento.pdf \
       --pdf-engine=xelatex \
       -V geometry:margin=2cm \
       -V lang:es
```

### Opción C: Con VS Code
1. Instala la extensión **Markdown PDF** (yzane.markdown-pdf).  
2. Clic derecho sobre el archivo → "Markdown PDF: Export (pdf)".

### Opción D: Convertir a .docx con Pandoc
```bash
pandoc docs/pipeline-tensores-y-entrenamiento.md \
       -o docs/pipeline-tensores-y-entrenamiento.docx
```

---

*Documento generado para el proyecto [The Color of Emotions](https://github.com/AcSsalazar/the-color-of-emotions). Redacción en español, estilo didáctico.*
