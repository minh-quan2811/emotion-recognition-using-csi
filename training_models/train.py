import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from model.models import initialize_models
from prepare_training_dataset.training_dataset_processing import custom_collate_fn

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))

    loaded_train_dataset = torch.load(os.path.join(repo_root, "processing_data", "my_video_csi_dataset", "train_dataset.pt"))
    loaded_val_dataset = torch.load(os.path.join(repo_root, "processing_data", "my_video_csi_dataset", "val_dataset.pt"))

    train_dataloader = DataLoader(loaded_train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(loaded_val_dataset, batch_size=32, collate_fn=custom_collate_fn)

    # Initialize models
    teacher_model, student_model = initialize_models()

    # Train
    train_losses, val_losses = train_model(student_model, teacher_model, train_dataloader, val_dataloader, 
                                        epochs=20, model_path=os.path.join(repo_root, "model_weights", "training_weights"))
