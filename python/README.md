# Bike vs Car CNN Model - Python Backend

Complete CNN model implementation with GPU support for bike vs car classification.

## 🚀 Features

- **GPU-Accelerated Training**: Uses TensorFlow/Keras with CUDA support
- **Custom CNN Architecture**: 4 convolutional blocks with batch normalization
- **Data Augmentation**: Rotation, zoom, shift, flip for robust training
- **Training Pipeline**: Complete with callbacks, logging, and visualization
- **REST API**: Flask API for inference and predictions
- **Model Checkpointing**: Auto-saves best model during training
- **Comprehensive Logging**: TensorBoard, CSV logs, and plots

## 📋 Requirements

- Python 3.8+
- CUDA Toolkit 11.2+ (for GPU support)
- cuDNN 8.1+ (for GPU support)

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Create the following directory structure:

```
data/
├── train/
│   ├── bike/
│   │   ├── img1.jpg
│   │   └── ...
│   └── car/
│       ├── img1.jpg
│       └── ...
├── validation/
│   ├── bike/
│   └── car/
└── test/
    ├── bike/
    └── car/
```

Run dataset setup:

```bash
python dataset.py
```

### 3. Verify GPU

Check if GPU is available:

```bash
python model.py
```

Expected output:
```
✓ GPU is available! Found 1 GPU(s):
  - /physical_device:GPU:0
```

## 🏋️ Training

### Basic Training

```bash
python train.py
```

### Custom Training Configuration

```bash
python train.py --epochs 100 --batch_size 32 --learning_rate 0.001 --img_size 224
```

### Training Parameters

- `--data_dir`: Data directory (default: 'data')
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--img_size`: Image size (default: 224)

### Training Output

- **Models**: Saved in `models/`
  - `best_model.keras`: Best model based on validation accuracy
  - `final_model.keras`: Final model after all epochs
  - `checkpoint_epoch_XX.keras`: Per-epoch checkpoints

- **Logs**: Saved in `logs/`
  - `training_log_YYYYMMDD_HHMMSS.csv`: Training metrics per epoch
  - `training_history.png`: Visualization of training curves
  - `training_config.json`: Configuration used for training
  - `test_results.json`: Final test set evaluation
  - `tensorboard_YYYYMMDD_HHMMSS/`: TensorBoard logs

### Monitoring Training

Use TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006

## 🔮 Inference

### Start API Server

```bash
python api.py
```

Server runs on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Single Image Prediction
```bash
POST /api/predict
Content-Type: multipart/form-data

Body: image file
```

Response:
```json
{
  "prediction": "Car",
  "confidence": 0.9876,
  "probabilities": {
    "Bike": 0.0124,
    "Car": 0.9876
  }
}
```

#### 3. Batch Prediction
```bash
POST /api/batch-predict
Content-Type: application/json

Body: {
  "images": ["base64_image1", "base64_image2", ...]
}
```

#### 4. Model Information
```bash
GET /api/model/info
```

#### 5. Training Metrics
```bash
GET /api/model/metrics
```

## 📊 Model Architecture

```
Input (224x224x3)
    ↓
Conv Block 1 (32 filters)
    ↓
Conv Block 2 (64 filters)
    ↓
Conv Block 3 (128 filters)
    ↓
Conv Block 4 (256 filters)
    ↓
Dense (512 units)
    ↓
Dense (256 units)
    ↓
Output (2 classes)
```

Each Conv Block includes:
- 2x Convolutional layers
- Batch Normalization
- MaxPooling (2x2)
- Dropout (0.25)

## 🎯 Performance Tips

### GPU Memory

If you encounter OOM errors, reduce batch size:
```bash
python train.py --batch_size 16
```

### Training Speed

- Use GPU for 10-50x faster training
- Increase batch size (if memory allows): `--batch_size 64`
- Use mixed precision training (add to model.py):
  ```python
  from tensorflow.keras import mixed_precision
  mixed_precision.set_global_policy('mixed_float16')
  ```

### Model Accuracy

- Increase training data (500+ images per class)
- More epochs: `--epochs 100`
- Data augmentation (already enabled)
- Fine-tune hyperparameters

## 🐛 Troubleshooting

### GPU Not Detected

1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
2. Install cuDNN: https://developer.nvidia.com/cudnn
3. Verify installation:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

### Out of Memory

- Reduce batch size: `--batch_size 16`
- Reduce image size: `--img_size 128`
- Close other GPU applications

### Model Not Found

Ensure you've trained the model first:
```bash
python train.py
```

The API looks for `models/best_model.keras`

## 📦 Model Export

### TensorFlow Serving

Export model for TensorFlow Serving:
```python
from model import BikeCarCNN

model = BikeCarCNN()
model.load_model('models/best_model.keras')
model.model.export('exported_model/')
```

### TensorFlow Lite (Mobile)

Convert to TFLite for mobile deployment:
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### ONNX (Cross-platform)

Export to ONNX format:
```bash
pip install tf2onnx
python -m tf2onnx.convert --keras models/best_model.keras --output model.onnx
```

## 📄 License

This project is ready for deployment and production use.

## 🤝 Support

For issues or questions, refer to:
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/
- Flask Documentation: https://flask.palletsprojects.com/
