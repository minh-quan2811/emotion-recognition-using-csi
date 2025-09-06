import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from model.models import initialize_models
from prepare_training_dataset.training_dataset_processing import custom_collate_fn


class CSIEmotionTrainer:
    def __init__(self, model, teacher_model, train_dataloader, val_dataloader, 
                 lr=0.001, milestones=[10, 80], gamma=0.1, model_path='./zmodel'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.teacher_model = teacher_model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_path = model_path

        self.mse_loss = nn.MSELoss()

        # Optimizer + Scheduler
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.best_vloss = float('inf')
        self.epoch_number = 0

    def feature_loss(self, teacher_features, student_features):
        return self.mse_loss(teacher_features[-1].detach(), student_features[-1])

    def train_one_epoch(self, epoch_index):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (video_frames, csi_data, one_hot_labels) in enumerate(self.train_dataloader):
            video_frames = video_frames.to(self.device)
            csi_data = csi_data.to(self.device)
            one_hot_labels = one_hot_labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                # Teacher forward pass
                teacher_outputs = self.teacher_model(video_frames, output_hidden_states=True)
                teacher_features = teacher_outputs.hidden_states

            with autocast():
                # Student forward pass
                student_logits, student_features = self.model(csi_data)

                logit_loss = F.cross_entropy(student_logits, one_hot_labels.argmax(dim=1))
                feat_loss = self.feature_loss(teacher_features, student_features) * 0.001

                loss = logit_loss + feat_loss

            # Print batch loss
            print(
                f"Batch {batch_idx + 1}, "
                f"Logit Loss: {logit_loss.item():.4f}, "
                f"Feature Loss: {feat_loss.item():.4f}, "
                f"Total Loss: {loss.item():.4f}"
            )

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_dataloader)

    def validate(self):
        self.model.eval()
        running_vloss = 0.0

        with torch.no_grad():
            for batch_idx, (video_frames, csi_data, one_hot_labels) in enumerate(self.val_dataloader):
                video_frames = video_frames.to(self.device)
                csi_data = csi_data.to(self.device)
                one_hot_labels = one_hot_labels.to(self.device)

                teacher_outputs = self.teacher_model(video_frames, output_hidden_states=True)
                teacher_features = teacher_outputs.hidden_states

                student_logits, student_features = self.model(csi_data)

                logit_loss = F.cross_entropy(student_logits, one_hot_labels.argmax(dim=1))
                feat_loss = self.feature_loss(teacher_features, student_features) * 0.001

                val_loss = logit_loss + feat_loss
                running_vloss += val_loss.item()

        return running_vloss / (batch_idx + 1)

    def train(self, epochs=20):
        for epoch in range(epochs):
            print(f"EPOCH {epoch + 1}:")

            avg_train_loss = self.train_one_epoch(epoch)
            self.train_losses.append(avg_train_loss)

            self.scheduler.step()

            avg_vloss = self.validate()
            self.val_losses.append(avg_vloss)

            print(f"LOSS train {avg_train_loss:.4f} valid {avg_vloss:.4f}")

            if avg_vloss < self.best_vloss:
                self.best_vloss = avg_vloss
                checkpoint_path = f'{self.model_path}_epoch_{epoch + 1}'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved new best model at epoch {epoch + 1} with validation loss {self.best_vloss:.4f}")

            self.epoch_number += 1

        return self.train_losses, self.val_losses


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))

    loaded_train_dataset = torch.load(os.path.join(repo_root, "processing_data", "my_video_csi_dataset", "train_dataset.pt"))
    loaded_val_dataset = torch.load(os.path.join(repo_root, "processing_data", "my_video_csi_dataset", "val_dataset.pt"))

    train_dataloader = DataLoader(loaded_train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(loaded_val_dataset, batch_size=32, collate_fn=custom_collate_fn)

    # Initialize models
    teacher_model, student_model = initialize_models()

    # Create Trainer instance
    trainer = CSIEmotionTrainer(student_model, teacher_model, train_dataloader, val_dataloader,
                                model_path=os.path.join(repo_root, "model_weights", "training_weights"))

    # Train
    train_losses, val_losses = trainer.train(epochs=20)
