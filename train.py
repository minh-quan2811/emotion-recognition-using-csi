import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import VisionTransformer, initialize_models
from data_processing import VideoCSIDataset, custom_collate_fn
from transformers import AutoFeatureExtractor

def compute_metrics(predictions, labels):
    # Convert predictions to class indices
    _, pred_classes = torch.max(predictions, dim=1)
    _, true_classes = torch.max(labels, dim=1)

    # Precision, Recall, F1-Score (use average='weighted' for multi-class)
    precision = precision_score(true_classes.cpu(), pred_classes.cpu(), average='weighted', zero_division=0)
    recall = recall_score(true_classes.cpu(), pred_classes.cpu(), average='weighted', zero_division=0)
    f1 = f1_score(true_classes.cpu(), pred_classes.cpu(), average='weighted', zero_division=0)

    return precision, recall, f1

def plot_confusion_matrix(predictions, labels, class_names):
    # Convert predictions to class indices
    _, pred_classes = torch.max(predictions, dim=1)
    _, true_classes = torch.max(labels, dim=1)

    cm = confusion_matrix(true_classes.cpu(), pred_classes.cpu())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def evaluate_student_model(model, test_dataloader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (video_frames, csi_data, one_hot_labels) in enumerate(test_dataloader):
            csi_data = csi_data.to(device)
            one_hot_labels = one_hot_labels.to(device)

            logits, _ = model(csi_data)

            # Compute precision, recall, f1
            precision, recall, f1 = compute_metrics(logits, one_hot_labels)
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Collect all predictions and labels for confusion matrix
            all_predictions.append(logits)
            all_labels.append(one_hot_labels)

    avg_precision = total_precision / (batch_idx + 1)
    avg_recall = total_recall / (batch_idx + 1)
    avg_f1 = total_f1 / (batch_idx + 1)

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Confusion matrix
    plot_confusion_matrix(all_predictions, all_labels, class_names)

    return avg_precision, avg_recall, avg_f1

def train_model(model, teacher_model, train_dataloader, val_dataloader, epochs=20, model_path='/content/zmodel'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_loss = nn.MSELoss()

    # Feature loss
    def feature_loss(teacher_features, student_features):
        return mse_loss(teacher_features[-1].detach(), student_features[-1])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 80], gamma=0.1)

    train_losses = []
    val_losses = []
    best_vloss = float('inf')
    epoch_number = 0

    def train_one_epoch(epoch_index):
        model.train()
        running_loss = 0.0
        for batch_idx, (video_frames, csi_data, one_hot_labels) in enumerate(train_dataloader):
            video_frames = video_frames.to(device)
            csi_data = csi_data.to(device)
            one_hot_labels = one_hot_labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                # Teacher model inference
                teacher_outputs = teacher_model(video_frames, output_hidden_states=True)
                teacher_features = teacher_outputs.hidden_states

            with autocast():
                # Student model inference
                student_logits, student_features = model(csi_data)

                # Cross-entropy loss for logits
                logit_loss = F.cross_entropy(student_logits, one_hot_labels.argmax(dim=1))

                # Feature distillation loss
                feat_loss = feature_loss(teacher_features, student_features)
                feat_loss = feat_loss * 0.001

                # Total loss
                loss = logit_loss + feat_loss

            # Print individual loss each batch
            print(f"Batch {batch_idx + 1}, Logit Loss: {logit_loss.item():.4f}, Feature Loss: {feat_loss.item():.4f}, Total Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average training loss
        avg_train_loss = running_loss / len(train_dataloader)
        return avg_train_loss

    # Training loop
    for epoch in range(epochs):
        print(f'EPOCH {epoch + 1}:')

        avg_train_loss = train_one_epoch(epoch)
        train_losses.append(avg_train_loss)

        scheduler.step()

        # Validation
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for batch_idx, (video_frames, csi_data, one_hot_labels) in enumerate(val_dataloader):
                video_frames = video_frames.to(device)
                csi_data = csi_data.to(device)
                one_hot_labels = one_hot_labels.to(device)

                # Teacher model inference
                teacher_outputs = teacher_model(video_frames, output_hidden_states=True)
                teacher_features = teacher_outputs.hidden_states

                # Student model inference
                student_logits, student_features = model(csi_data)

                # Cross-entropy loss for logits
                logit_loss = F.cross_entropy(student_logits, one_hot_labels.argmax(dim=1))

                # Feature distillation loss
                feat_loss = feature_loss(teacher_features, student_features)
                feat_loss = feat_loss * 0.001

                # Total loss
                val_loss = logit_loss + feat_loss
                running_vloss += val_loss.item()

        avg_vloss = running_vloss / (batch_idx + 1)
        val_losses.append(avg_vloss)

        print(f'LOSS train {avg_train_loss:.4f} valid {avg_vloss:.4f}')

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            checkpoint_path = f'{model_path}_epoch_{epoch + 1}'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model at epoch {epoch + 1} with validation loss {best_vloss:.4f}")

        epoch_number += 1

    return train_losses, val_losses

if __name__ == "__main__":
    # Configuration
    model_name = "dima806/facial_emotions_image_detection"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    segment_length = 600
    step_size = 400
    root_dir = "/content/drive/MyDrive/Quan Emotions"
    class_names = ['Happy', 'Sad', 'Neutral', 'Angry']

    # Dataset and DataLoader
    dataset = VideoCSIDataset(root_dir, feature_extractor, segment_length=segment_length, step_size=step_size)
    total_size = len(dataset)
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=custom_collate_fn)

    # Initialize models
    teacher_model, student_model = initialize_models()

    # Train
    train_losses, val_losses = train_model(student_model, teacher_model, train_dataloader, val_dataloader, epochs=20)

    # Evaluate
    student_model.eval()
    avg_precision, avg_recall, avg_f1 = evaluate_student_model(student_model, val_dataloader, class_names)
    print(f"Average Precision: {avg_precision}, Recall: {avg_recall}, F1: {avg_f1}")
