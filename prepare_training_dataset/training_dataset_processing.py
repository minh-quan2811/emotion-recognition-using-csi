import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import cv2
from PIL import Image
import torchvision.transforms as transforms
from scipy.signal import spectrogram
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor


# Global label map for emotions
label_map = {'Happy': 0, 'Sad': 1, 'Neutral': 2, 'Angry': 3}


class VideoCSIDataset(Dataset):
    """PyTorch Dataset for synchronized Video + CSI emotion data."""

    def __init__(self, root_dir, feature_extractor, transform=None,
                 segment_length=None, step_size=None, fs=10000):
        self.root_dir = root_dir
        self.transform = transform
        self.segment_length = segment_length
        self.step_size = step_size
        self.feature_extractor = feature_extractor
        self.fs = fs
        self.mtcnn = MTCNN(image_size=224, margin=0)

        self.data_segments = self._load_dataset()

    def __len__(self):
        return len(self.data_segments)

    def __getitem__(self, idx):
        video_segment, csi_segment, label = self.data_segments[idx]

        # One-hot encoding [1,0,0,0] -> happy, [0,1,0,0] -> sad
        one_hot_label = torch.zeros(len(label_map))
        one_hot_label[label] = 1

        # Process CSI spectrogram
        A = self._process_csi_data([csi_segment])
        spectrogram = self._process_A_to_spectrograms(A)[0]
        normalized_spectrogram = self._normalize_spectrogram(spectrogram)

        return video_segment, normalized_spectrogram, one_hot_label

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _normalize_spectrogram(self, spectrograms):
        means = spectrograms.mean(dim=(-2, -1), keepdim=True)
        stds = spectrograms.std(dim=(-2, -1), keepdim=True)
        return (spectrograms - means) / stds

    def _process_csi_data(self, csi_segments):
        csi_data_list = np.stack(csi_segments, axis=0)
        A = np.sqrt(csi_data_list[:, :, ::2]**2 + csi_data_list[:, :, 1::2]**2)
        A = np.transpose(A, (0, 2, 1))
        return A

    def _process_A_to_spectrograms(self, A):
        num_segments, num_features = A.shape[0], A.shape[1]
        spectrograms = np.zeros((num_segments, num_features, 224, 224))

        for segment_idx in range(num_segments):
            for feature_idx in range(num_features):
                f, t, Sxx = spectrogram(A[segment_idx, feature_idx, :], nperseg=200)
                Sxx = np.where(Sxx > 0, Sxx, 1e-10)
                Sxx = 10 * np.log10(Sxx)
                Sxx_resized = cv2.resize(Sxx, (224, 224), interpolation=cv2.INTER_LINEAR)
                spectrograms[segment_idx, feature_idx, :, :] = torch.from_numpy(Sxx_resized).float()

        subcarriers_to_remove = list(range(0, 6)) + [32] + list(range(59, 64))
        subcarriers_to_keep = [i for i in range(64) if i not in subcarriers_to_remove]
        spectrograms = spectrograms[:, subcarriers_to_keep, :, :]

        return torch.FloatTensor(spectrograms)

    def _segment_csi_data(self, csi_data):
        segments = []
        for i in range(0, len(csi_data) - self.segment_length + 1, self.step_size):
            segment = csi_data[i:i + self.segment_length]
            segments.append(segment)
        return segments

    def _get_video_segments(self, video_path, num_segments):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))    # Video frame rate
        step_size_in_seconds = self.step_size / self.fs     # Step size in seconds

        video_segments = []
        for segment_idx in range(num_segments):
            start_second = segment_idx * step_size_in_seconds      # Start time for this segment
            frame = self._get_frame_at_time(cap, start_second, fps)
            if frame is not None:
                video_segments.append(frame)

        cap.release()
        return video_segments

    def _get_frame_at_time(self, cap, start_second, fps):
        start_frame = int(start_second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            return None
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        to_pil = transforms.ToPILImage()
        img_cropped = self.mtcnn(frame)

        if img_cropped is not None:
            img_cropped_pil = (img_cropped + 1) / 2
            img_cropped_pil = to_pil(img_cropped_pil)
            frame = self.feature_extractor(images=img_cropped_pil, return_tensors="pt")["pixel_values"]
        else:
            frame = self.feature_extractor(images=frame, return_tensors="pt")["pixel_values"]

        return frame

    def _load_dataset(self):
        data_segments = []
        for emotion in os.listdir(self.root_dir):
            emotion_dir = os.path.join(self.root_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue

            video_dir = os.path.join(emotion_dir, f'teacher_data_{emotion.lower()}')
            csi_dir = os.path.join(emotion_dir, f'student_data_{emotion.lower()}')

            video_files = sorted(os.listdir(video_dir))
            csi_files = sorted(os.listdir(csi_dir))

            for video_file, csi_file in zip(video_files, csi_files):
                if video_file.endswith('.mp4') and csi_file.endswith('.csv'):
                    video_path = os.path.join(video_dir, video_file)
                    csi_path = os.path.join(csi_dir, csi_file)
                    # print(csi_path)
                    # print(video_path)

                    csi_data = pd.read_csv(csi_path, usecols=[25], header=None, skiprows=1)
                    csi_data = csi_data.dropna(subset=[25])
                    csi_data = csi_data[25].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=' '))
                    csi_data = np.stack(csi_data.values)
                    csi_segments = self._segment_csi_data(csi_data)

                    num_csi_segments = len(csi_segments)
                    video_segments = self._get_video_segments(video_path, num_csi_segments)

                    for video_segment, csi_segment in zip(video_segments, csi_segments):
                        data_segments.append((video_segment, csi_segment, label_map[emotion]))
        return data_segments


class VideoCSIDatasetBuilder:
    """Handles dataset splitting and saving."""
    @staticmethod
    def save_dataset_splits(train_dataset, val_dataset, test_dataset, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        print(f"Ensured directory '{save_folder}' exists.")

        paths = {
            "train": os.path.join(save_folder, "train_dataset.pt"),
            "val": os.path.join(save_folder, "val_dataset.pt"),
            "test": os.path.join(save_folder, "test_dataset.pt"),
        }

        try:
            torch.save(train_dataset, paths["train"])
            print(f"Train dataset saved successfully to: {paths['train']}")
            torch.save(val_dataset, paths["val"])
            print(f"Validation dataset saved successfully to: {paths['val']}")
            torch.save(test_dataset, paths["test"])
            print(f"Test dataset saved successfully to: {paths['test']}")
        except Exception as e:
            print(f"An error occurred while saving datasets: {e}")
            for p in paths.values():
                if os.path.exists(p): os.remove(p)
            raise


def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    video_segments = [item[0] for item in batch]
    spectrograms = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])

    video_padded = torch.cat(video_segments, dim=0) if video_segments else None
    return video_padded, spectrograms, labels


if __name__ == "__main__":
    # Teacher model config
    model_name = "dima806/facial_emotions_image_detection"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # Dataset params
    segment_length = 600
    step_size = 400

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))
    root_dir = os.path.join(repo_root, "Emotions dataset")

    # Dataset + splits
    dataset = VideoCSIDataset(root_dir, feature_extractor, segment_length=segment_length, step_size=step_size)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Save datasets
    destination_folder = "my_video_csi_dataset"
    VideoCSIDatasetBuilder.save_dataset_splits(train_dataset, val_dataset, test_dataset, destination_folder)