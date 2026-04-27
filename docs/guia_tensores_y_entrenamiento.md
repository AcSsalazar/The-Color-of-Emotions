# Guía del Pipeline de Tensores y Entrenamiento CNN

> **Tono didáctico en español.** Esta guía cubre el pipeline completo del notebook 3.2: desde la carga del audio crudo hasta el entrenamiento de una CNN con los packs `.pt` generados.

---

## 1. Objetivo del pipeline

El objetivo es transformar archivos de audio (`.wav`) en **tensores tridimensionales** con forma `[3, N_MELS, TARGET_FRAMES]` listos para entrenar una red neuronal convolucional (CNN).

Cada tensor agrupa **tres canales**:

| Canal | Feature | Descripción |
|-------|---------|-------------|
| 0 | Mel-Spectrogram (dB) | Representación de energía por banda de frecuencia perceptual |
| 1 | Delta (orden 1) | Derivada temporal del mel-spectrogram (dinámica) |
| 2 | Cochleagram | Representación inspirada en la percepción auditiva humana |

Al final del pipeline, los tensores de cada split se empaquetan en un único archivo `.pt` con metadatos portables (sin rutas absolutas).

---

## 2. Orden del preprocesamiento

El pipeline aplica las siguientes etapas **en este orden exacto**. El orden no es arbitrario: cada paso deja la señal en el estado correcto para el siguiente.

```
Audio crudo (.wav)
    │
    ▼
[1] Resample → SAMPLE_RATE = 16 000 Hz
    │
    ▼
[2] Trim basado en energía  (eliminar silencios)
    │
    ▼
[3] Normalización de amplitud (pico → −1..+1)
    │
    ▼
[4] Padding / Truncado a TARGET_SAMPLES
    │
    ▼
[5] Extracción de features → tensor [3, N_MELS, frames]
    │
    ▼
[6] Ajuste de frames (fix_frames) → tensor [3, N_MELS, TARGET_FRAMES]
    │
    ▼
[7] Z-score por muestra y por canal → tensor [3, N_MELS, TARGET_FRAMES]
```

### 2.1 Resample

Si el archivo viene a una tasa distinta (ej. 44 100 Hz en RAVDESS), se remuestrea a **16 000 Hz** (límite del corpus CREMA-D) antes de cualquier otra operación.

```python
if waveform.shape[0] != SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, orig_freq, SAMPLE_RATE)
```

### 2.2 Trim por energía

Se eliminan los silencios de cabeza y cola usando un umbral dinámico equivalente a **−40 dB**:

$$20 \cdot \log_{10}(0.01) = -40\,\text{dB}$$

```python
mse = torch.mean(waveform ** 2)
if mse > 1e-6:
    threshold = torch.max(torch.abs(waveform)) * 0.01   # −40 dB equivalente
    mask = torch.abs(waveform[0]) > threshold
    idx  = torch.where(mask)[0]
    if len(idx) > 0:
        waveform = waveform[:, idx[0]:idx[-1] + 1]
```

**¿Por qué −40 dB y no un valor fijo?**
El umbral se calcula como fracción del pico de esa señal concreta (`max_abs * 0.01`), lo que hace que sea **dinámico**: si una grabación es muy suave, el umbral también lo es, y así no se recorta contenido útil. Este comportamiento replica el `top_db=40` de Librosa pero en PyTorch.

La condición `mse > 1e-6` protege señales casi silenciosas: si la señal entera es ruido de fondo, el trim no se aplica.

### 2.3 Normalización de amplitud (pico)

Tras el trim, la forma de onda se escala para que su valor pico absoluto sea 1:

```python
peak = torch.max(torch.abs(waveform))
if peak > 0:
    waveform = waveform / peak
```

**¿Por qué aquí y no al final?**
- Estandariza la amplitud **antes** de calcular el espectrograma, de modo que el rango de dB del mel-spectrogram sea consistente entre grabaciones.
- Si normalizaras *después* del espectrograma, estarías modificando magnitudes logarítmicas de forma no uniforme.

### 2.4 Padding / Truncado a `TARGET_SAMPLES`

La duración objetivo es **1.6 s**, lo que corresponde exactamente a 50 frames con `HOP_LENGTH = 512`:

$$MAX\_DURATION = \frac{TARGET\_FRAMES \times HOP\_LENGTH}{SAMPLE\_RATE} = \frac{50 \times 512}{16\,000} = 1.6\,\text{s}$$

- Si la señal es **más corta**: se aplica *zero-padding* centrado.
- Si la señal es **más larga**: se trunca desde el centro.

```python
TARGET_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)   # 25 600 muestras

def fix_length(waveform, target):
    n = waveform.shape[-1]
    if n < target:
        pad = target - n
        left = pad // 2
        right = pad - left
        waveform = torch.nn.functional.pad(waveform, (left, right))
    elif n > target:
        start = (n - target) // 2
        waveform = waveform[:, start:start + target]
    return waveform
```

**Justificación de 1.6 s vs 3.0 s:** en la versión anterior del pipeline (`MAX_DURATION = 3.0 s`) se generaban ~94 frames, pero la inspección visual del dataset mostró que la información acústica relevante se concentraba entre los frames 30 y 50. Reducir a 1.6 s elimina el silencio redundante y hace la entrada más compacta y eficiente.

---

## 3. Extracción de features (mel / delta / cochleagram)

El extractor `AudioFeatureExtractor` produce el tensor `[3, N_MELS, frames]`:

```python
class AudioFeatureExtractor(torch.nn.Module):
    def __init__(self, sr=16000, n_mels=60, hop=512):
        super().__init__()
        self.mel  = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sr, n_mels=n_mels, hop_length=hop)
        self.db   = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.coch = nnAudio.features.CQT(sr=sr, hop_length=hop,
                                          n_bins=n_mels, bins_per_octave=12)

    def forward(self, waveform):
        # Canal 0: Mel en dB
        mel_db  = self.db(self.mel(waveform))            # [1, n_mels, T]
        # Canal 1: Delta (derivada temporal orden 1)
        delta   = torchaudio.functional.compute_deltas(mel_db)
        # Canal 2: Cochleagram (CQT via nnAudio)
        coch    = torch.abs(self.coch(waveform))         # [1, n_bins, T]
        coch_db = self.db(coch)
        return torch.cat([mel_db, delta, coch_db], dim=0)  # [3, n_mels, T]
```

---

## 4. Z-score por muestra y por canal

### 4.1 ¿Por qué z-score si ya normalizamos la amplitud?

La normalización de amplitud (paso 2.3) actúa sobre la **forma de onda en el tiempo** y garantiza que el pico sea ±1. Sin embargo, después de extraer los features:

- El **mel-spectrogram en dB** puede tener un rango de, por ejemplo, −80 a 0 dB.
- El **delta** oscila alrededor de 0 pero con escala diferente.
- El **cochleagram** tiene su propia distribución.

Cada canal vive en una **distribución diferente**. Si no normalizas, la CNN "ve" los tres canales con magnitudes muy distintas, lo que:
1. Introduce sesgos hacia el canal con mayor varianza.
2. Dificulta la convergencia del optimizador (gradientes desbalanceados).
3. Puede saturar funciones de activación (ReLU, etc.).

El z-score **centra en 0 y escala a varianza 1 cada canal de cada muestra**, igualando las distribuciones de entrada.

### 4.2 Por muestra y por canal (no por dataset)

La normalización se aplica **por muestra individual** (no sobre la media/std del dataset completo):

$$x_{\text{norm}}[c, :, :] = \frac{x[c, :, :] - \mu_c}{\sigma_c + \varepsilon}$$

donde $\mu_c$ y $\sigma_c$ son la media y desviación estándar del canal $c$ **de esa muestra**.

**Ventajas:**
- No hay *data leakage*: no necesitas calcular estadísticas del conjunto de entrenamiento antes de procesar.
- Robusto a grabaciones con muy diferente nivel de energía.
- Funciona bien en inferencia con muestras nuevas.

**Código:**

```python
def zscore_per_sample_per_channel(tensor, eps=1e-6):
    """
    tensor: [3, N_MELS, T]
    Aplica z-score canal a canal para una sola muestra.
    """
    # tensor.shape = [C, H, W]
    mean = tensor.mean(dim=(-2, -1), keepdim=True)   # [C, 1, 1]
    std  = tensor.std(dim=(-2, -1), keepdim=True)    # [C, 1, 1]
    return (tensor - mean) / (std + eps)
```

### 4.3 ¿Por qué z-score *después* de `fix_frames`?

`fix_frames` ajusta el número de columnas (frames) del tensor mediante padding o recorte:

```python
def fix_frames(tensor, target_frames):
    """tensor: [C, H, W] → [C, H, target_frames]"""
    T = tensor.shape[-1]
    if T < target_frames:
        pad = target_frames - T
        tensor = torch.nn.functional.pad(tensor, (0, pad))
    else:
        tensor = tensor[..., :target_frames]
    return tensor
```

Si aplicas z-score **antes** de `fix_frames`, el padding de ceros que se añade después modifica la distribución del canal (introduce ceros extra que bajan la media y reducen la std). Al aplicarlo **después**, las estadísticas reflejan exactamente el contenido real del espectrograma ya con su forma final.

**Orden correcto:**

```
extract_features → [3, N_MELS, T_variable]
fix_frames       → [3, N_MELS, TARGET_FRAMES]   ← padding/recorte aquí
z-score          → [3, N_MELS, TARGET_FRAMES]   ← normalizar sobre el tensor completo
```

---

## 5. Data augmentation fijo

El augmentation se aplica **solo en el split `train`** y **solo para la clase `surprised`**, porque es la clase minoritaria en el dataset combinado RAVDESS + CREMA-D.

Se generan **dos versiones adicionales** por archivo original, en el dominio de la onda (antes de extraer features):

```python
import torch

def add_noise(waveform, snr_db=30):
    """Ruido AWGN gaussiano."""
    signal_power = waveform.pow(2).mean()
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return waveform + noise

def time_shift(waveform, shift_frac=0.1):
    """Desplazamiento temporal aleatorio ±10 % de la duración."""
    n = waveform.shape[-1]
    shift = int(n * shift_frac * (2 * torch.rand(1).item() - 1))
    return torch.roll(waveform, shifts=shift, dims=-1)
```

Cada versión se registra en la metadata con el campo `"version"`:

| `version` | Descripción |
|-----------|-------------|
| `"original"` | Audio sin modificar |
| `"noise"` | AWGN leve (SNR ≈ 30 dB) |
| `"shift"` | Desplazamiento temporal ±10 % |

Esto permite auditar en cualquier momento cuántos aumentos se generaron y garantizar que `val`/`test` no contienen versiones sintéticas.

---

## 6. Efecto en Matplotlib (`vmin` / `vmax`)

Cuando visualizas un tensor con `imshow` o `matshow` de Matplotlib, los parámetros `vmin` y `vmax` controlan el rango de colores del colormap.

### 6.1 Antes del z-score

El mel-spectrogram en dB tiene valores típicamente en el rango **[−80, 0]** dB:

```python
import matplotlib.pyplot as plt

# Tensor canal 0 sin z-score: rango −80..0 dB
plt.imshow(tensor[0], aspect="auto", origin="lower",
           cmap="magma", vmin=-80, vmax=0)
plt.colorbar(label="dB")
plt.title("Mel-Spectrogram (antes del z-score)")
plt.show()
```

### 6.2 Después del z-score

Tras el z-score, los valores están centrados en 0 con desviación estándar ≈ 1. El rango efectivo suele ser **[−3, +3]**:

```python
# Tensor canal 0 tras z-score: rango ≈ −3..+3
plt.imshow(tensor_norm[0], aspect="auto", origin="lower",
           cmap="RdBu_r", vmin=-3, vmax=3)
plt.colorbar(label="z-score")
plt.title("Mel-Spectrogram (después del z-score)")
plt.show()
```

**¿Qué cambia visualmente?**

| Aspecto | Antes del z-score | Después del z-score |
|---------|-------------------|---------------------|
| Rango de valores | −80 .. 0 dB | ≈ −3 .. +3 |
| Colormap recomendado | `magma`, `inferno` (unipolar) | `RdBu_r`, `bwr` (bipolar) |
| `vmin` / `vmax` sugeridos | `−80`, `0` | `−3`, `3` |
| Información visible | Energía absoluta | Contraste relativo dentro de cada muestra |

> **Tip:** Si no estableces `vmin`/`vmax`, Matplotlib usa el mínimo y máximo del tensor. Tras el z-score esto funciona razonablemente, pero explicitarlos da visualizaciones comparables entre muestras.

### 6.3 Visualizar los tres canales

```python
def plot_tensor(tensor, title="Tensor"):
    """
    tensor: [3, N_MELS, TARGET_FRAMES] ya normalizado con z-score.
    """
    channel_names = ["Mel-Spectrogram", "Delta", "Cochleagram"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(tensor[i].numpy(), aspect="auto", origin="lower",
                       cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title(f"Canal {i}: {channel_names[i]}")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")
        plt.colorbar(im, ax=ax, label="z-score")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
```

---

## 7. Entrenamiento con los packs `.pt`

### 7.1 Estructura del pack

Cada split se guarda como un `dict` en un único archivo `.pt`:

```python
{
    "x":            torch.Tensor,    # [N, 3, N_MELS, TARGET_FRAMES]
    "y":            torch.LongTensor,# [N]
    "meta":         list[dict],      # largo N, sin rutas absolutas
    "class_to_idx": dict,            # {"angry": 0, "happy": 1, ...}
    "config":       dict             # sr, hop, n_mels, target_frames, etc.
}
```

### 7.2 Cargar los packs

```python
import torch

train_pack = torch.load("train_pack.pt", weights_only=False)
val_pack   = torch.load("val_pack.pt",   weights_only=False)
test_pack  = torch.load("test_pack.pt",  weights_only=False)

X_train, y_train = train_pack["x"], train_pack["y"]
X_val,   y_val   = val_pack["x"],   val_pack["y"]
X_test,  y_test  = test_pack["x"],  test_pack["y"]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

### 7.3 Dataset y DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class PackedTensorDataset(Dataset):
    def __init__(self, pack):
        self.x    = pack["x"]
        self.y    = pack["y"]
        self.meta = pack["meta"]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# Instanciar
train_ds = PackedTensorDataset(train_pack)
val_ds   = PackedTensorDataset(val_pack)
test_ds  = PackedTensorDataset(test_pack)

# DataLoaders
BATCH_SIZE = 32

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)
```

### 7.4 Arquitecturas CNN recomendadas

#### A. Baseline CNN (punto de partida)

Pequeña y fácil de depurar. Úsala primero para verificar que el pipeline funciona.

```python
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, n_classes=8, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),                                        # H/2, W/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                                        # H/4, W/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4)),                          # 128×4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

#### B. ResNet adaptado

Usa `torchvision.models.resnet18` con la primera capa adaptada a 3 canales (ya coincide) y la capa final ajustada al número de clases.

```python
import torchvision.models as models

def build_resnet18(n_classes=8, pretrained=True):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    # La entrada ya es [B, 3, H, W] → no se necesita cambiar conv1
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
```

> **Tip:** Con transfer learning desde ImageNet el modelo converge más rápido incluso en espectrogramas, porque los primeros filtros detectan bordes/texturas que también son útiles en mapas de frecuencia-tiempo.

#### C. EfficientNet-B0

Mejor balance precisión/parámetros que ResNet para datasets pequeños (~9 k muestras).

```python
def build_efficientnet_b0(n_classes=8, pretrained=True):
    model = models.efficientnet_b0(
        weights="IMAGENET1K_V1" if pretrained else None)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, n_classes)
    return model
```

### 7.5 Regularización

| Técnica | Dónde | Valor sugerido |
|---------|-------|----------------|
| Dropout | Antes de la capa final | 0.3 – 0.5 |
| BatchNorm | Después de cada Conv | — |
| Weight decay (L2) | Optimizador | 1e-4 |
| Label smoothing | `CrossEntropyLoss` | 0.1 |
| Early stopping | Basado en val loss | paciencia = 10 epochs |

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

### 7.6 Métricas recomendadas

Dado que el dataset está **desequilibrado** (surprise < angry < happy < …), la *accuracy* global puede ser engañosa. Usa:

```python
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

acc    = Accuracy(task="multiclass", num_classes=8, average="macro")
f1     = F1Score(task="multiclass",  num_classes=8, average="macro")
cm     = ConfusionMatrix(task="multiclass", num_classes=8)
```

### 7.7 Hiperparámetros de partida

| Hiperparámetro | Valor sugerido |
|----------------|----------------|
| `batch_size` | 32 |
| `lr` inicial | 1e-3 (AdamW) |
| `weight_decay` | 1e-4 |
| Épocas máximas | 100 |
| `label_smoothing` | 0.1 |
| Scheduler | CosineAnnealingLR, T_max=50 |
| Paciencia early stop | 10 epochs (val loss) |

### 7.8 Bucle de entrenamiento mínimo

```python
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = build_resnet18(n_classes=8).to(device)

best_val_loss = float("inf")
patience_counter = 0
PATIENCE = 10

for epoch in range(1, 101):
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch:03d} | "
          f"Train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
          f"Val loss {va_loss:.4f} acc {va_acc:.3f}")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break
```

---

## 8. Checklist de sanidad (*sanity check*)

Antes de lanzar el entrenamiento, verifica cada punto:

- [ ] **Shapes uniformes**: todos los tensores en el pack tienen forma `[3, N_MELS, TARGET_FRAMES]`.
  ```python
  shapes = set(tuple(x.shape) for x in X_train)
  assert len(shapes) == 1, f"Shapes inconsistentes: {shapes}"
  ```

- [ ] **Sin NaN ni Inf**:
  ```python
  assert not torch.isnan(X_train).any(), "NaN en X_train"
  assert not torch.isinf(X_train).any(), "Inf en X_train"
  ```

- [ ] **Z-score correcto** (media ≈ 0, std ≈ 1 por canal por muestra):
  ```python
  sample = X_train[0]          # [3, N_MELS, T]
  print(sample.mean(dim=(-2,-1)))   # debe ser ≈ [0, 0, 0]
  print(sample.std(dim=(-2,-1)))    # debe ser ≈ [1, 1, 1]
  ```

- [ ] **Distribución de clases** (detectar desequilibrio):
  ```python
  import pandas as pd
  pd.Series(y_train.numpy()).value_counts().sort_index()
  ```

- [ ] **Augmentation solo en train**: comprobar que `val` y `test` no contienen `version != "original"`:
  ```python
  for m in val_pack["meta"] + test_pack["meta"]:
      assert m["version"] == "original", f"Versión sintética en val/test: {m}"
  ```

- [ ] **`class_to_idx` consistente** entre splits:
  ```python
  assert train_pack["class_to_idx"] == val_pack["class_to_idx"] == test_pack["class_to_idx"]
  ```

- [ ] **Visualización de un batch**: ejecuta `plot_tensor(X_train[0])` y verifica que los tres canales tienen aspecto coherente (no son matrices de ceros o ruido puro).

- [ ] **Forward pass sin error**:
  ```python
  model.eval()
  with torch.no_grad():
      out = model(X_train[:4].to(device))
  print(out.shape)  # debe ser [4, n_classes]
  ```

---

*Guía generada en el contexto del proyecto **The Color of Emotions** — pipeline 3.2.*
