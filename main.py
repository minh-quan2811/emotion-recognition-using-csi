"""
Main entrypoint for CSI Emotion Recognition Model using argparse

Usage examples:
    python main.py prepare --data-dir <path> --output-dir <path>
    python main.py train --dataset-dir <path> --output-dir <path>
    python main.py test --model-path <path> --dataset-dir <path>
    python main.py predict --model-path <path> --csi-file <path>
"""

import os
import sys
import argparse
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


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_dataset(args):
    print('=' * 60)
    print('DATASET PREPARATION')
    print('=' * 60)

    print(f"\nLoading feature extractor: {args.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    print(f"\nLoading dataset from: {args.data_dir}")
    print(f"Segment length: {args.segment_length}")
    print(f"Step size: {args.step_size}")

    dataset = VideoCSIDataset(
        args.data_dir,
        feature_extractor,
        segment_length=args.segment_length,
        step_size=args.step_size
    )

    print(f"\nTotal samples: {len(dataset)}")

    total_size = len(dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size

    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {test_size}")

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"\nSaving datasets to: {args.output_dir}")
    VideoCSIDatasetBuilder.save_dataset_splits(
        train_dataset, val_dataset, test_dataset, args.output_dir
    )

    print('\nDataset preparation complete!')
    print('=' * 60)


def train_model(args):
    print('=' * 60)
    print('MODEL TRAINING')
    print('=' * 60)

    train_dataset_path = os.path.join(args.dataset_dir, 'train_dataset.pt')
    val_dataset_path = os.path.join(args.dataset_dir, 'val_dataset.pt')

    if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path):
        print("Error: Dataset files not found. Please run 'prepare' command first.")
        sys.exit(1)

    print(f"\nLoading datasets from: {args.dataset_dir}")
    loaded_train_dataset = torch.load(train_dataset_path)
    loaded_val_dataset = torch.load(val_dataset_path)

    print(f"Train samples: {len(loaded_train_dataset)}")
    print(f"Validation samples: {len(loaded_val_dataset)}")

    train_dataloader = DataLoader(
        loaded_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        loaded_val_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn
    )

    print('\nInitializing models...')
    teacher_model, student_model = initialize_models()

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        student_model.load_state_dict(torch.load(args.resume, map_location=get_device()))

    print('\nTraining configuration:')
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Milestones: {args.milestones}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Output directory: {args.output_dir}")

    trainer = CSIEmotionTrainer(
        student_model,
        teacher_model,
        train_dataloader,
        val_dataloader,
        lr=args.lr,
        milestones=args.milestones,
        gamma=args.gamma,
        model_path=os.path.join(args.output_dir, 'model')
    )

    print('\nStarting training...')
    print('=' * 60)
    train_losses, val_losses = trainer.train(epochs=args.epochs)

    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {trainer.best_vloss:.4f}")


def load_student_model(model_path, device):
    print(f"\nLoading model from: {model_path}")
    config = ViTConfig(num_channels=52, num_labels=4, output_hidden_states=True)
    student_model = VisionTransformer(config).to(device)
    state = torch.load(model_path, map_location=device)
    student_model.load_state_dict(state)
    student_model.eval()
    return student_model


def evaluate_dataset(args):
    print('=' * 60)
    print('MODEL EVALUATION')
    print('=' * 60)

    device = get_device()
    student_model = load_student_model(args.model_path, device)

    dataset_path = os.path.join(args.dataset_dir, f"{args.dataset_type}_dataset.pt")

    print(f"Loading {args.dataset_type} dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    loaded_dataset = torch.load(dataset_path)
    dataloader = DataLoader(
        loaded_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn
    )

    print(f"Total samples: {len(loaded_dataset)}")
    print('\nEvaluating model...')
    print('=' * 60)
    class_names = ['Happy', 'Sad', 'Neutral', 'Angry']
    avg_precision, avg_recall, avg_f1 = evaluate_student_model(
        student_model, dataloader, class_names
    )

    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")


def predict_file(args):
    print('=' * 60)
    print('SINGLE FILE PREDICTION')
    print('=' * 60)

    device = get_device()
    student_model = load_student_model(args.model_path, device)

    print(f"Processing CSI file: {args.csi_file}")
    class_names = ['Happy', 'Sad', 'Neutral', 'Angry']

    predicted_labels, emotion_counts = test_single_file(
        args.csi_file,
        student_model,
        class_names,
        segment_length=args.segment_length,
        step_size=args.step_size
    )

    print('\n' + '=' * 60)
    print('PREDICTION RESULTS')
    print('=' * 60)
    print(f"\nTotal segments: {len(predicted_labels)}")
    print('\nEmotion distribution:')
    for emotion, count in emotion_counts.items():
        percentage = (count / len(predicted_labels)) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")

    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
    print(f"\nDominant emotion: {dominant_emotion[0]}")


def build_parser():
    parser = argparse.ArgumentParser(description='CSI Emotion Recognition')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')

    # prepare dataset
    p_prepare = subparsers.add_parser('prepare', help='Prepare and split the dataset')
    p_prepare.add_argument('--data-dir', required=True, help='Path to raw emotions dataset directory')
    p_prepare.add_argument('--output-dir', default='./my_video_csi_dataset', help='Output directory for processed datasets')
    p_prepare.add_argument('--model-name', default='dima806/facial_emotions_image_detection', help='Pretrained model name for feature extraction')
    p_prepare.add_argument('--segment-length', default=600, type=int, help='CSI segment length')
    p_prepare.add_argument('--step-size', default=400, type=int, help='Step size for segmentation')
    p_prepare.add_argument('--train-split', default=0.8, type=float, help='Training set ratio')
    p_prepare.add_argument('--val-split', default=0.1, type=float, help='Validation set ratio')
    p_prepare.set_defaults(func=prepare_dataset)

    # train
    p_train = subparsers.add_parser('train', help='Train the student model')
    p_train.add_argument('--dataset-dir', required=True, help='Directory containing prepared datasets')
    p_train.add_argument('--output-dir', default='./model_weights', help='Output directory for model checkpoints')
    p_train.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    p_train.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    p_train.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    p_train.add_argument('--milestones', nargs='+', type=int, default=[10, 80], help='Learning rate scheduler milestones')
    p_train.add_argument('--gamma', default=0.1, type=float, help='Learning rate decay factor')
    p_train.add_argument('--resume', default=None, help='Path to checkpoint to resume training')
    p_train.set_defaults(func=train_model)

    # test
    p_test = subparsers.add_parser('test', help='Test model on dataset')
    p_test.add_argument('--model-path', required=True, help='Path to trained model weights')
    p_test.add_argument('--dataset-dir', required=True, help='Directory containing prepared datasets')
    p_test.add_argument('--dataset-type', choices=['test'], default='test', help='Dataset type to evaluate on')
    p_test.add_argument('--batch-size', default=32, type=int, help='Batch size for evaluation')
    p_test.set_defaults(func=evaluate_dataset)

    # predict
    p_predict = subparsers.add_parser('predict', help='Predict emotion from a single CSI file')
    p_predict.add_argument('--model-path', required=True, help='Path to trained model weights')
    p_predict.add_argument('--csi-file', required=True, help='Path to CSI CSV file')
    p_predict.add_argument('--segment-length', default=600, type=int, help='CSI segment length')
    p_predict.add_argument('--step-size', default=400, type=int, help='Step size for segmentation')
    p_predict.set_defaults(func=predict_file)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == '__main__':
    main()
