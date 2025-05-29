from collections import Counter
import pandas as pd
import numpy as np
import torch
from model import VisionTransformer
from data_processing import normalize_spectrogram, segment_csi_data, process_csi_data, process_A_to_spectrograms
from transformers import ViTConfig

def evaluate_single_csi_file(csi_file, model, class_names, segment_length=600, step_size=400):
    # Load CSI data
    csi_data = pd.read_csv(csi_file, usecols=[25], header=None, skiprows=1)
    csi_data = csi_data.dropna(subset=[25])
    csi_data = csi_data[25].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=' '))
    csi_data = np.stack(csi_data.values)

    # Segment CSI data
    csi_segments = segment_csi_data(csi_data, segment_length, step_size)

    # Convert to spectrogram
    A = process_csi_data(csi_segments)
    spectrograms = process_A_to_spectrograms(A)
    spectrograms = normalize_spectrogram(spectrograms)

    spectrograms = spectrograms.cuda()

    # Make predictions
    with torch.no_grad():
        logits, _ = model(spectrograms)
        _, predictions = torch.max(logits, dim=1)

    predicted_labels = [class_names[pred.item()] for pred in predictions]

    # Count the number of each emotion
    emotion_counts = Counter(predicted_labels)

    return predicted_labels, emotion_counts

if __name__ == "__main__":
    # Configuration
    config = ViTConfig(num_channels=52, num_labels=4, output_hidden_states=True)
    class_names = ['Happy', 'Sad', 'Neutral', 'Angry']
    csi_file_path = "/content/drive/MyDrive/neutraltest.csv"

    # Load model
    student_model = VisionTransformer(config).cuda()
    student_model.load_state_dict(torch.load('/content/zmodel_epoch_8'))
    student_model.eval()

    # Evaluate
    predicted_labels, emotion_counts = evaluate_single_csi_file(csi_file_path, student_model, class_names)
    print("Predicted emotions:", predicted_labels)
    print("Emotion counts:", emotion_counts)