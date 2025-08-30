import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from transformers import ViTConfig
from torch.utils.data import DataLoader

from model.models import VisionTransformer
from processing_data.training_dataset_processing import custom_collate_fn

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

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))

    model_weights_dir = os.path.join(repo_root, "model_weights", "model8")

    # Load student model
    config = ViTConfig(num_channels=52, num_labels=4, output_hidden_states=True)
    student_model = VisionTransformer(config).cuda()
    student_model.load_state_dict(torch.load(model_weights_dir))
    student_model.eval()

    # Load DataLoader
    loaded_val_dataset = torch.load(os.path.join(repo_root,"processing_data" , "my_video_csi_dataset", "val_dataset.pt"))
    loaded_test_dataset = torch.load(os.path.join(repo_root,"processing_data" , "my_video_csi_dataset", "test_dataset.pt"))

    val_dataloader = DataLoader(loaded_val_dataset, batch_size=32, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(loaded_test_dataset, batch_size=32, collate_fn=custom_collate_fn)

    # Evaluate on validation/test set
    student_model.eval()
    class_names = ['Happy', 'Sad', 'Neutral', 'Angry']
    avg_precision, avg_recall, avg_f1 = evaluate_student_model(student_model, val_dataloader, class_names)
    print(f"Average Precision: {avg_precision}, Recall: {avg_recall}, F1: {avg_f1}")