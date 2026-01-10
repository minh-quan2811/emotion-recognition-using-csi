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
- **argparse** - command-line interface (used in `main.py`)

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
pip install -r requirements.txt
```

### 2. Prepare Dataset
Process raw emotion data and create train/val/test splits:
```bash
python main.py prepare --data-dir "./Emotions dataset" --output-dir ./processing_data/my_video_csi_dataset
```

**Options:**
- `--data-dir`: Path to raw emotions dataset (required)
- `--output-dir`: Output directory for processed datasets (default: `./my_video_csi_dataset`)
- `--segment-length`: CSI segment length (default: 600)
- `--step-size`: Step size for segmentation (default: 400)
- `--train-split`: Training set ratio (default: 0.8)
- `--val-split`: Validation set ratio (default: 0.1)

### 3. Train Model
Train the student model with knowledge distillation:
```bash
python main.py train --dataset-dir ./processing_data/my_video_csi_dataset --output-dir ./model_weights --epochs 20
```

**Options:**
- `--dataset-dir`: Directory containing prepared datasets (required)
- `--output-dir`: Output directory for model checkpoints (default: `./model_weights`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--milestones`: LR scheduler milestones (default: 10, 80)
- `--gamma`: Learning rate decay factor (default: 0.1)
- `--resume`: Path to checkpoint to resume training (optional)

### 4. Evaluate Model
Evaluate model performance on test/validation datasets:
```bash
python main.py test --model-path ./model_weights/model_epoch_15 --dataset-dir ./processing_data/my_video_csi_dataset
```

**Options:**
- `--model-path`: Path to trained model weights (required)
- `--dataset-dir`: Directory containing prepared datasets (required)
- `--dataset-type`: Dataset to evaluate on: `train`, `val`, or `test` (default: `test`)
- `--batch-size`: Batch size for evaluation (default: 32)

### 5. Predict on New Data
Predict emotion from a single CSI file:
```bash
python main.py predict --model-path ./model_weights/model_epoch_15 --csi-file ./sample_data/angry7.csv
```

**Options:**
- `--model-path`: Path to trained model weights (required)
- `--csi-file`: Path to CSI CSV file (required)
- `--segment-length`: CSI segment length (default: 600)
- `--step-size`: Step size for segmentation (default: 400)

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

