# The Color of Emotions: Speech Emotion Recognition (SER)

> **Analysis and implementation of neural networks trained on multimodal data (*early fusion*), numerical data (3D tensors), and 2D images.**

---

##  Abstract
This project addresses the differences in training neural networks fed with data derived from audio signals. These signals are transformed into two distinct formats: multidimensional numerical data using PyTorch and two-dimensional images generated from their Mel spectrograms.

Throughout the chapters of this project, fundamental concepts of signal processing, pattern recognition, and feature extraction are explored, while also integrating an artistic perspective of sound conceived as image.

The main objective of this repository is to create a clean and technically accurate data collection, available in two formats:

- [**2D Images:**](https://drive.google.com/file/d/16sWxbncvc9iazebQPa6vCrHe6KZkgUQS/view?usp=sharing) Mel spectrograms and MFCC in PNG format (224x224 resolution).
- [**3D Tensors:**](https://drive.google.com/drive/folders/1GIuh9wlE3dxeL5lUlKGAkK66VWKCJU9L?usp=sharing) Based on MFCC coefficients and their first and second order deltas.

---

## Chapters and Topics Covered

1. **Preliminary analysis:** Exploration of acoustic descriptors with Librosa, audio sample playback.
2. **Feature extraction:** Obtaining, visualizing, and explaining key acoustic descriptors (MFCC, Mel spectrograms).
3. **Multimodal dataset creation:** Generation of visual and matrix representations for model training.
4. **Training:** Training neural networks using two types of available data: *computer vision* vs *audio processing*.
5. **Multimodal training:** Evaluation of efficiency and accuracy of reinforced trained models (ResNet18 vs EfficientNet-B0).
6. **Results analysis:** Evaluation of the best-performing model and visualization of emotional patterns.
7. **Artistic perspective:** Mapping frequency intensities to color spaces to generate representative visual palettes for each emotion.

---

##  Tools and Technologies

### Audio Processing
- **Librosa:** Audio analysis and Mel spectrogram generation.
- **NumPy:** Numerical computation and tensor manipulation.

### Image Processing and Visualization
- **Matplotlib / Seaborn / Plotly:** Spectrogram visualization and statistical analysis.
- **Pillow (PIL) / Scikit-image:** Advanced image manipulation and processing.
- **ColorThief:** Color palette extraction.

### Machine Learning & Deep Learning
- **Scikit-learn:** Pattern recognition and *clustering* algorithms.
- **TensorFlow / PyTorch:** Building and training *Deep Learning* models for classification.

---

## Methodology and Project Phases

### Phase 1: Audio Processing and Extraction
- Loading emotional sound samples.
- Extraction of MFCC coefficients and generation of Mel spectrograms.
- Data augmentation (noise, speed variation) and *dataset* combination to increase diversity in the final dataset.
- Data normalization and preprocessing.

### Phase 2: Data Split and Final Exploratory Analysis
- Statistical analysis to identify frequency patterns.
- Class *clustering* among the data.
- Extraction of key features by emotion.

### Phase 3: Color Extraction and Artistic Vision
- Mapping frequency intensities to color spaces (RGB, HSV, LAB).
- Generation of specific color palettes (5-10 colors per emotion) and temporal gradients.
- Design of aesthetic backgrounds and export of metadata (PNG, HEX, JSON).

---

##  Research Questions

The development of this project seeks to answer the following questions:
1. Are there consistent frequency patterns for specific emotions in different sounds?
2. How efficient is training *Deep Learning* models using numerical data (3D tensors) versus images (spectrograms)?
3. Is there room for improvement in classification after implementing a reinforcement method using multimodal data with *early fusion*?
4. Is it possible to generate color palettes that visually represent emotions based on their frequency characteristics?
5. Can these color palettes be used in practical applications such as UI/UX design, music visualization, or generative art?

---

##  Practical Applications

- **UI/UX Design:** Color schemes for interfaces based on emotional awareness.
- **Real-time detection:** Identification of emotions in real-time to detect mood changes.
- **Generative Art:** Spatial ambience schemes based on obtained color palettes.

---

##  Repository Structure

*(Directory structure pending update as the project progresses).*

---

##  How to Contribute
Contribution guidelines (sound sample submissions, quality standards for palettes and code) will be available soon.

##  License
[To be defined]

##  Contact
**Project Maintenance:** acsalazar-19@hotmail.com 
**Creation Date:** March 11, 2026  
**Status:** Draft - Phase Two
