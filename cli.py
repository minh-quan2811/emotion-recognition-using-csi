"""
CLI for CSI Emotion Recognition Model

Usage:
    python cli.py prepare --data-dir <path> --output-dir <path>
    python cli.py train --dataset-dir <path> --output-dir <path>
    python cli.py test --model-path <path> --dataset-dir <path>
    python cli.py predict --model-path <path> --csi-file <path>
"""

import os
import sys
import click
import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, ViTConfig

from model.models import initialize_models, VisionTransformer
from prepare_training_dataset.training_dataset_processing import (
    VideoCSIDataset, VideoCSIDatasetBuilder, custom_collate_fn
)
from training_models.train import CSIEmotionTrainer
from testing_models.student_model_eval_mectrics import evaluate_student_model
from testing_models.test_student_model import evaluate_student_model as test_single_file


@click.group()
def cli():
    """CSI Emotion Recognition Model CLI"""
    pass


@cli.command()
@click.option('--data-dir', required=True, type=click.Path(exists=True),
              help='Path to raw emotions dataset directory')
@click.option('--output-dir', default='./my_video_csi_dataset',
              help='Output directory for processed datasets')
@click.option('--model-name', default='dima806/facial_emotions_image_detection',
              help='Pretrained model name for feature extraction')
@click.option('--segment-length', default=600, type=int,
              help='CSI segment length')
@click.option('--step-size', default=400, type=int,
              help='Step size for segmentation')
@click.option('--train-split', default=0.8, type=float,
              help='Training set ratio')
@click.option('--val-split', default=0.1, type=float,
              help='Validation set ratio')
def prepare(data_dir, output_dir, model_name, segment_length, step_size, train_split, val_split):
    """Prepare and split the dataset"""
    click.echo("=" * 60)
    click.echo("DATASET PREPARATION")
    click.echo("=" * 60)
    
    click.echo(f"\nLoading feature extractor: {model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    click.echo(f"\nLoading dataset from: {data_dir}")
    click.echo(f"Segment length: {segment_length}")
    click.echo(f"Step size: {step_size}")
    
    dataset = VideoCSIDataset(
        data_dir,
        feature_extractor,
        segment_length=segment_length,
        step_size=step_size
    )
    
    click.echo(f"\nTotal samples: {len(dataset)}")
    
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    click.echo(f"Train samples: {train_size}")
    click.echo(f"Validation samples: {val_size}")
    click.echo(f"Test samples: {test_size}")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    click.echo(f"\nSaving datasets to: {output_dir}")
    VideoCSIDatasetBuilder.save_dataset_splits(
        train_dataset, val_dataset, test_dataset, output_dir
    )
    
    click.secho("\nDataset preparation complete!", fg='green', bold=True)
    click.echo("=" * 60)


@cli.command()
@click.option('--dataset-dir', required=True, type=click.Path(exists=True),
              help='Directory containing prepared datasets')
@click.option('--output-dir', default='./model_weights',
              help='Output directory for model checkpoints')
@click.option('--epochs', default=20, type=int,
              help='Number of training epochs')
@click.option('--batch-size', default=32, type=int,
              help='Batch size for training')
@click.option('--lr', default=0.001, type=float,
              help='Learning rate')
@click.option('--milestones', default=[10, 80], multiple=True, type=int,
              help='Learning rate scheduler milestones')
@click.option('--gamma', default=0.1, type=float,
              help='Learning rate decay factor')
@click.option('--resume', default=None, type=click.Path(exists=True),
              help='Path to checkpoint to resume training')
def train(dataset_dir, output_dir, epochs, batch_size, lr, milestones, gamma, resume):
    """Train the student model"""
    click.echo("=" * 60)
    click.echo("MODEL TRAINING")
    click.echo("=" * 60)
    
    train_dataset_path = os.path.join(dataset_dir, "train_dataset.pt")
    val_dataset_path = os.path.join(dataset_dir, "val_dataset.pt")
    
    if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path):
        click.secho("Error: Dataset files not found. Please run 'prepare' command first.", 
                    fg='red', bold=True)
        sys.exit(1)
    
    click.echo(f"\nLoading datasets from: {dataset_dir}")
    loaded_train_dataset = torch.load(train_dataset_path)
    loaded_val_dataset = torch.load(val_dataset_path)
    
    click.echo(f"Train samples: {len(loaded_train_dataset)}")
    click.echo(f"Validation samples: {len(loaded_val_dataset)}")
    
    train_dataloader = DataLoader(
        loaded_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        loaded_val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    
    click.echo("\nInitializing models...")
    teacher_model, student_model = initialize_models()
    
    if resume:
        click.echo(f"Resuming from checkpoint: {resume}")
        student_model.load_state_dict(torch.load(resume))
    
    click.echo(f"\nTraining configuration:")
    click.echo(f"  Epochs: {epochs}")
    click.echo(f"  Learning rate: {lr}")
    click.echo(f"  Batch size: {batch_size}")
    click.echo(f"  Milestones: {list(milestones)}")
    click.echo(f"  Gamma: {gamma}")
    click.echo(f"  Output directory: {output_dir}")
    
    trainer = CSIEmotionTrainer(
        student_model,
        teacher_model,
        train_dataloader,
        val_dataloader,
        lr=lr,
        milestones=list(milestones),
        gamma=gamma,
        model_path=os.path.join(output_dir, "model")
    )
    
    click.echo("\nStarting training...")
    click.echo("=" * 60)
    train_losses, val_losses = trainer.train(epochs=epochs)
    
    click.echo("\n" + "=" * 60)
    click.secho("TRAINING COMPLETE", fg='green', bold=True)
    click.echo("=" * 60)
    click.echo(f"Final training loss: {train_losses[-1]:.4f}")
    click.echo(f"Final validation loss: {val_losses[-1]:.4f}")
    click.echo(f"Best validation loss: {trainer.best_vloss:.4f}")


@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True),
              help='Path to trained model weights')
@click.option('--dataset-dir', required=True, type=click.Path(exists=True),
              help='Directory containing prepared datasets')
@click.option('--dataset-type', type=click.Choice(['train', 'val', 'test']), default='test',
              help='Dataset type to evaluate on')
@click.option('--batch-size', default=32, type=int,
              help='Batch size for evaluation')
def test(model_path, dataset_dir, dataset_type, batch_size):
    """Test model on dataset"""
    click.echo("=" * 60)
    click.echo("MODEL EVALUATION")
    click.echo("=" * 60)
    
    click.echo(f"\nLoading model from: {model_path}")
    config = ViTConfig(num_channels=52, num_labels=4, output_hidden_states=True)
    student_model = VisionTransformer(config).cuda()
    student_model.load_state_dict(torch.load(model_path))
    student_model.eval()
    
    dataset_path = os.path.join(dataset_dir, f"{dataset_type}_dataset.pt")
    
    click.echo(f"Loading {dataset_type} dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        click.secho(f"Error: Dataset file not found: {dataset_path}", fg='red', bold=True)
        sys.exit(1)
    
    loaded_dataset = torch.load(dataset_path)
    dataloader = DataLoader(
        loaded_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn
    )
    
    click.echo(f"Total samples: {len(loaded_dataset)}")
    
    click.echo("\nEvaluating model...")
    click.echo("=" * 60)
    class_names = ['Happy', 'Sad', 'Neutral', 'Angry']
    avg_precision, avg_recall, avg_f1 = evaluate_student_model(
        student_model, dataloader, class_names
    )
    
    click.echo("\n" + "=" * 60)
    click.secho("EVALUATION RESULTS", fg='green', bold=True)
    click.echo("=" * 60)
    click.echo(f"Average Precision: {avg_precision:.4f}")
    click.echo(f"Average Recall: {avg_recall:.4f}")
    click.echo(f"Average F1-Score: {avg_f1:.4f}")


@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True),
              help='Path to trained model weights')
@click.option('--csi-file', required=True, type=click.Path(exists=True),
              help='Path to CSI CSV file')
@click.option('--segment-length', default=600, type=int,
              help='CSI segment length')
@click.option('--step-size', default=400, type=int,
              help='Step size for segmentation')
def predict(model_path, csi_file, segment_length, step_size):
    """Predict emotion from a single CSI file"""
    click.echo("=" * 60)
    click.echo("SINGLE FILE PREDICTION")
    click.echo("=" * 60)
    
    click.echo(f"\nLoading model from: {model_path}")
    config = ViTConfig(num_channels=52, num_labels=4, output_hidden_states=True)
    student_model = VisionTransformer(config).cuda()
    student_model.load_state_dict(torch.load(model_path))
    student_model.eval()
    
    click.echo(f"Processing CSI file: {csi_file}")
    class_names = ['Happy', 'Sad', 'Neutral', 'Angry']
    
    predicted_labels, emotion_counts = test_single_file(
        csi_file,
        student_model,
        class_names,
        segment_length=segment_length,
        step_size=step_size
    )
    
    click.echo("\n" + "=" * 60)
    click.secho("PREDICTION RESULTS", fg='green', bold=True)
    click.echo("=" * 60)
    click.echo(f"\nTotal segments: {len(predicted_labels)}")
    click.echo("\nEmotion distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(predicted_labels)) * 100
        click.echo(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
    click.secho(f"\nDominant emotion: {dominant_emotion[0]}", fg='cyan', bold=True)


if __name__ == '__main__':
    cli()