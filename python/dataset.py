"""
Dataset handling and preprocessing utilities for Bike vs Car classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
from pathlib import Path


class BikeCarDataset:
    """Handle dataset loading, preprocessing, and augmentation"""
    
    def __init__(self, data_dir='data', img_size=(224, 224), batch_size=32):
        """
        Initialize dataset handler
        
        Args:
            data_dir: Root directory containing train/val/test folders
            img_size: Target image dimensions (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['Bike', 'Car']
        
    def create_data_augmentation(self):
        """
        Create image data augmentation for training
        
        Augmentation includes:
        - Rotation (±20 degrees)
        - Width/Height shifts (±20%)
        - Zoom (±20%)
        - Horizontal flip
        - Brightness adjustment
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation/Test data should only be rescaled
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        return train_datagen, val_datagen
    
    def load_data_generators(self):
        """
        Load data using ImageDataGenerator
        
        Expected directory structure:
        data/
          train/
            bike/
              img1.jpg
              img2.jpg
            car/
              img1.jpg
              img2.jpg
          validation/
            bike/
            car/
          test/
            bike/
            car/
        
        Returns:
            train_generator, val_generator, test_generator
        """
        train_datagen, val_datagen = self.create_data_augmentation()
        
        # Training data
        train_dir = self.data_dir / 'train'
        if train_dir.exists():
            train_generator = train_datagen.flow_from_directory(
                str(train_dir),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=True
            )
        else:
            print(f"⚠ Training directory not found: {train_dir}")
            train_generator = None
        
        # Validation data
        val_dir = self.data_dir / 'validation'
        if val_dir.exists():
            val_generator = val_datagen.flow_from_directory(
                str(val_dir),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=False
            )
        else:
            print(f"⚠ Validation directory not found: {val_dir}")
            val_generator = None
        
        # Test data
        test_dir = self.data_dir / 'test'
        if test_dir.exists():
            test_generator = val_datagen.flow_from_directory(
                str(test_dir),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=False
            )
        else:
            print(f"⚠ Test directory not found: {test_dir}")
            test_generator = None
        
        return train_generator, val_generator, test_generator
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to image file or PIL Image or numpy array
            
        Returns:
            preprocessed_image: Normalized image array
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, np.ndarray):
            img = Image.fromarray(image_path)
        else:
            img = image_path
        
        # Convert to RGB
        img = img.convert('RGB')
        
        # Resize
        img = img.resize(self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        return img_array
    
    def create_sample_dataset_structure(self):
        """
        Create sample dataset directory structure
        This helps users understand the expected format
        """
        directories = [
            self.data_dir / 'train' / 'bike',
            self.data_dir / 'train' / 'car',
            self.data_dir / 'validation' / 'bike',
            self.data_dir / 'validation' / 'car',
            self.data_dir / 'test' / 'bike',
            self.data_dir / 'test' / 'car',
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created dataset directory structure at: {self.data_dir}")
        print("\nExpected structure:")
        print("data/")
        print("  ├── train/")
        print("  │   ├── bike/")
        print("  │   └── car/")
        print("  ├── validation/")
        print("  │   ├── bike/")
        print("  │   └── car/")
        print("  └── test/")
        print("      ├── bike/")
        print("      └── car/")
        print("\nPlace your images in the respective folders.")
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        info = {
            'train': self._count_images(self.data_dir / 'train'),
            'validation': self._count_images(self.data_dir / 'validation'),
            'test': self._count_images(self.data_dir / 'test')
        }
        return info
    
    def _count_images(self, directory):
        """Count images in a directory"""
        if not directory.exists():
            return {'bike': 0, 'car': 0, 'total': 0}
        
        counts = {}
        total = 0
        
        for class_name in self.class_names:
            class_dir = directory / class_name.lower()
            if class_dir.exists():
                count = len(list(class_dir.glob('*.[jp][pn]g')) + 
                          list(class_dir.glob('*.jpeg')))
                counts[class_name.lower()] = count
                total += count
            else:
                counts[class_name.lower()] = 0
        
        counts['total'] = total
        return counts


def download_sample_dataset():
    """
    Instructions for downloading sample bike vs car dataset
    """
    print("="*60)
    print("Sample Dataset Sources")
    print("="*60)
    print("\n1. Kaggle Dataset:")
    print("   https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset")
    print("\n2. Manual Collection:")
    print("   - Collect bike images from Google Images")
    print("   - Collect car images from Google Images")
    print("   - Ensure diverse lighting, angles, and backgrounds")
    print("\n3. Recommended split:")
    print("   - Training: 70-80% of images")
    print("   - Validation: 10-15% of images")
    print("   - Test: 10-15% of images")
    print("\n4. Minimum images per class:")
    print("   - Training: 500+ images per class")
    print("   - Validation: 100+ images per class")
    print("   - Test: 100+ images per class")
    print("="*60)


if __name__ == "__main__":
    # Create dataset handler
    dataset = BikeCarDataset()
    
    # Create directory structure
    dataset.create_sample_dataset_structure()
    
    # Display dataset info
    print("\n" + "="*60)
    print("Dataset Information")
    print("="*60)
    info = dataset.get_dataset_info()
    for split, counts in info.items():
        print(f"\n{split.capitalize()}:")
        print(f"  Bike: {counts['bike']} images")
        print(f"  Car: {counts['car']} images")
        print(f"  Total: {counts['total']} images")
    
    # Show download instructions
    print("\n")
    download_sample_dataset()
