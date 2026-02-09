"""
Training script for Bike vs Car CNN Model
Optimized for GPU training with TensorFlow/Keras
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau, 
    TensorBoard,
    CSVLogger
)

from model import BikeCarCNN, check_gpu_availability
from dataset import BikeCarDataset


class CNNTrainer:
    """Handle complete training pipeline"""
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary with training parameters
        """
        self.config = config
        self.model_wrapper = BikeCarCNN(
            input_shape=tuple(config['input_shape']),
            num_classes=config['num_classes']
        )
        self.dataset = BikeCarDataset(
            data_dir=config['data_dir'],
            img_size=tuple(config['img_size']),
            batch_size=config['batch_size']
        )
        self.history = None
        
        # Create output directories
        self.models_dir = Path('models')
        self.logs_dir = Path('logs')
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def prepare_callbacks(self):
        """Prepare training callbacks"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=str(self.models_dir / 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Save checkpoints every epoch
            ModelCheckpoint(
                filepath=str(self.models_dir / f'checkpoint_epoch_{{epoch:02d}}.keras'),
                save_freq='epoch',
                verbose=0
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=str(self.logs_dir / f'tensorboard_{timestamp}'),
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV logging
            CSVLogger(
                filename=str(self.logs_dir / f'training_log_{timestamp}.csv'),
                separator=',',
                append=False
            )
        ]
        
        return callbacks
    
    def train(self):
        """Execute complete training pipeline"""
        print("="*80)
        print("Starting Training Pipeline")
        print("="*80)
        
        # Check GPU
        gpu_available = check_gpu_availability()
        
        # Load data
        print("\n[1/5] Loading dataset...")
        train_gen, val_gen, test_gen = self.dataset.load_data_generators()
        
        if train_gen is None:
            print("❌ No training data found!")
            print("Please prepare your dataset first using dataset.py")
            return
        
        print(f"✓ Training samples: {train_gen.samples}")
        if val_gen:
            print(f"✓ Validation samples: {val_gen.samples}")
        
        # Build and compile model
        print("\n[2/5] Building model...")
        self.model_wrapper.build_model()
        self.model_wrapper.compile_model(learning_rate=self.config['learning_rate'])
        print("✓ Model built and compiled")
        
        # Prepare callbacks
        print("\n[3/5] Preparing callbacks...")
        callbacks = self.prepare_callbacks()
        print("✓ Callbacks prepared")
        
        # Train model
        print("\n[4/5] Training model...")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("-"*80)
        
        self.history = self.model_wrapper.model.fit(
            train_gen,
            epochs=self.config['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        print("\n[5/5] Saving final model...")
        final_model_path = self.models_dir / 'final_model.keras'
        self.model_wrapper.save_model(str(final_model_path))
        
        # Evaluate on test set if available
        if test_gen:
            print("\n" + "="*80)
            print("Evaluating on Test Set")
            print("="*80)
            test_loss, test_acc, test_prec, test_recall = self.model_wrapper.model.evaluate(
                test_gen, 
                verbose=1
            )
            print(f"\nTest Results:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  Precision: {test_prec:.4f}")
            print(f"  Recall: {test_recall:.4f}")
            
            # Save test results
            test_results = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_acc),
                'test_precision': float(test_prec),
                'test_recall': float(test_recall)
            }
            
            with open(self.logs_dir / 'test_results.json', 'w') as f:
                json.dump(test_results, f, indent=2)
        
        # Plot training history
        self.plot_history()
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        print(f"Best model saved to: {self.models_dir / 'best_model.keras'}")
        print(f"Final model saved to: {final_model_path}")
        print(f"Training logs saved to: {self.logs_dir}")
        
    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        if 'val_accuracy' in self.history.history:
            axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        if 'val_loss' in self.history.history:
            axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train')
            if 'val_precision' in self.history.history:
                axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train')
            if 'val_recall' in self.history.history:
                axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.logs_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {self.logs_dir / 'training_history.png'}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Bike vs Car CNN Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'img_size': [args.img_size, args.img_size],
        'input_shape': [args.img_size, args.img_size, 3],
        'num_classes': 2,
        'early_stopping_patience': 10
    }
    
    # Save config
    with open('logs/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer and start training
    trainer = CNNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
