# emotion-recognition-using-csi
A research-based machine learning project that leverages WiFi Channel State Information (CSI) for emotion recognition through analysis of breathing-induced signal variations. This study introduces a non-invasive and privacy-preserving approach that replaces traditional image or wearable-based methods. By applying knowledge distillation from a Vision Transformer (ViT) trained on facial expressions to a CSI-based ViT model, the system effectively bridges the visual and wireless domains, achieving accurate emotion classification from spectrogram representations of CSI signals.

## Overview

This project uses knowledge distillation to transfer emotion recognition capabilities from video to WiFi signals:
- **Teacher Model**: Vision Transformer trained on facial expressions
- **Student Model**: Vision Transformer adapted for CSI spectrograms
- **Emotions**: Happy, Sad, Neutral, Angry

## Project Structure

```
.
├── model/
│   └── models.py                           # Model architecture
├── prepare_training_dataset/
│   └── training_dataset_processing.py      # Dataset preparation
├── training_models/
│   └── train.py                            # Training script
├── testing_models/
│   ├── student_model_eval_metrics.py       # Model evaluation
│   └── test_student_model.py               # Single file inference
├── model_weights/                          # Saved checkpoints
├── processing_data/                        # Processed datasets
└── Emotions dataset/                       # Raw data
```

## Tech Stack

### Frameworks & Libraries
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Vision Transformer models
- **facenet-pytorch** - Face detection (MTCNN)
- **OpenCV** - Video processing
- **SciPy** - Signal processing (spectrograms)
- **scikit-learn** - Evaluation metrics
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization

### Models
- **Teacher**: `dima806/facial_emotions_image_detection` (pre-trained ViT)
- **Student**: Custom ViT (52 channels for CSI, 4 output classes)

### Key Techniques
- Knowledge Distillation (logit + feature-based)
- Spectrogram generation from CSI data
- Face detection and cropping
- Mixed precision training

## Dataset Structure

```
Emotions dataset/
├── Happy/
│   ├── teacher_data_happy/      # MP4 videos
│   └── student_data_happy/      # CSV CSI files
├── Sad/
├── Neutral/
└── Angry/
```

The emotion recognition dataset is large (9.25GB) and stored externally.

You can download it here:  
[Emotions dataset - Google Drive] (https://drive.google.com/drive/folders/11euByOm9QnQgdzX6C3e-TukrJBf4qAIS)


## Quick Start

### 1. Install Dependencies
```bash
pip install requirements.txt
```

### 2. Prepare Dataset
```bash
cd prepare_training_dataset
python training_dataset_processing.py
```

### 3. Train Model
```bash
cd training_models
python train.py
```

### 4. Evaluate
```bash
cd testing_models
python student_model_eval_metrics.py
```

### 5. Test Single File
```bash
cd testing_models
python test_student_model.py
```

## Key Features

### Data Processing
- CSI to spectrogram conversion (224x224)
- Face detection and extraction from video
- Synchronized video-CSI pairing
- Train/val/test split (80%/10%/10%)

### Training
- Knowledge distillation loss (cross-entropy + MSE)
- Adam optimizer with MultiStepLR scheduling
- Mixed precision training
- Automatic best model checkpointing

### Evaluation
- Precision, Recall, F1-Score
- Confusion matrix visualization
- Per-segment and aggregated predictions

## Configuration

### Data Parameters
- Segment length: 600 samples
- Step size: 400 samples
- Sampling rate: 10kHz
- Image size: 224x224

### Training Parameters
- Batch size
- Learning rate
- Epochs
- LR decay
- Feature loss weight

## Model Architecture

### Teacher Model
- Input: RGB images (3 channels, 224x224)
- Architecture: Vision Transformer
- Output: 4 emotion classes

### Student Model
- Input: CSI spectrograms (52 channels, 224x224)
- Architecture: Vision Transformer (weight initialized from teacher)
- Output: 4 emotion classes + hidden states

### Loss Function
```
Total Loss = Cross-Entropy Loss + 0.001 × Feature MSE Loss
```

## Workflow

1. **Data Collection**: Synchronized video + CSI recordings
2. **Preprocessing**: Face extraction + CSI spectrogram generation
3. **Training**: Knowledge distillation from teacher to student
4. **Evaluation**: Metrics computation and visualization
5. **Inference**: Real-time emotion prediction from CSI

