"""
CNN Model Architecture for Bike vs Car Classification
Designed for GPU training with TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np


class BikeCarCNN:
    """CNN Model for binary classification: Bike vs Car"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize the CNN model
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of output classes (2 for bike/car)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build CNN architecture optimized for bike vs car classification
        
        Architecture:
        - 4 Convolutional blocks with increasing filters (32, 64, 128, 256)
        - MaxPooling after each conv block
        - Batch Normalization for stable training
        - Dropout for regularization
        - Dense layers for classification
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss, and metrics
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
    def get_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath='models/bike_car_cnn.keras'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/bike_car_cnn.keras'):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
    def predict(self, image):
        """
        Make prediction on a single image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            predictions: Array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
            
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        predictions = self.model.predict(image, verbose=0)
        return predictions
    
    def predict_class(self, image, class_names=['Bike', 'Car']):
        """
        Predict class label and confidence
        
        Args:
            image: Preprocessed image array
            class_names: List of class names
            
        Returns:
            class_name: Predicted class name
            confidence: Prediction confidence (0-1)
        """
        predictions = self.predict(image)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return class_names[class_idx], float(confidence)


def check_gpu_availability():
    """Check if GPU is available for TensorFlow"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ GPU is available! Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        
        # Enable memory growth to avoid OOM errors
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth setting error: {e}")
    else:
        print("⚠ No GPU found. Training will use CPU (slower).")
        print("To enable GPU:")
        print("  1. Install CUDA toolkit")
        print("  2. Install cuDNN")
        print("  3. Install tensorflow-gpu or use tensorflow>=2.15")
    
    return len(gpus) > 0


if __name__ == "__main__":
    # Check GPU availability
    check_gpu_availability()
    
    # Create and display model
    print("\n" + "="*60)
    print("Building Bike vs Car CNN Model")
    print("="*60)
    
    model = BikeCarCNN()
    model.build_model()
    model.compile_model()
    model.get_summary()
