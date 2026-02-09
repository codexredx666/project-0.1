# 🚀 Bike vs Car CNN Classifier

A complete end-to-end deep learning project for classifying images of bikes and cars using a custom CNN model. Built with TensorFlow/Keras for GPU-accelerated training and a React frontend for easy inference.

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.0+-red?logo=keras)
![React](https://img.shields.io/badge/React-18+-blue?logo=react)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)

## 🎯 Features

- **Custom CNN Architecture**: 4 convolutional blocks with batch normalization and dropout
- **GPU Accelerated**: Optimized for CUDA/cuDNN with TensorFlow/Keras
- **Data Augmentation**: Rotation, zoom, shift, flip for robust training
- **Production Ready**: Complete pipeline from training to deployment
- **REST API**: Flask API for model inference
- **Web Interface**: Beautiful React UI for image upload and predictions
- **Comprehensive Logging**: TensorBoard, CSV logs, and training visualizations
- **Model Checkpointing**: Auto-saves best model during training

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

### CNN Model

```
Input (224x224x3)
    ↓
Conv Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Conv Block 3: Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
    ↓
Conv Block 4: Conv2D(256) → BatchNorm → Conv2D(256) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → Dropout(0.5)
    ↓
Output: Dense(2, softmax) → [Bike, Car]
```

**Total Parameters**: ~7.8M trainable parameters

### Technology Stack

**Backend (Python)**:
- TensorFlow 2.15+ with GPU support
- Keras 3.0+ for model building
- Flask + Flask-CORS for API
- OpenCV + Pillow for image processing

**Frontend (React)**:
- React 18 with TypeScript
- Tailwind CSS for styling
- Lucide React for icons
- Sonner for notifications

## 🔧 Prerequisites

### System Requirements

- **GPU Training**: 
  - NVIDIA GPU with CUDA Compute Capability 3.5+
  - CUDA Toolkit 11.2+
  - cuDNN 8.1+
  - 4GB+ GPU memory (8GB+ recommended)

- **CPU Training** (slower):
  - Multi-core CPU (4+ cores recommended)
  - 8GB+ RAM

### Software Requirements

- Python 3.8+
- Node.js 16+ (for React frontend)
- pip or conda

## 📦 Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd bike-car-classifier
```

### 2. Install Python Dependencies

```bash
cd python
pip install -r requirements.txt
```

**For GPU Support**, ensure CUDA and cuDNN are installed:

```bash
# Verify GPU availability
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

### 3. Install Frontend Dependencies

```bash
# From project root
npm install
# or
bun install
```

## 📊 Dataset Preparation

### Expected Directory Structure

```
data/
├── train/
│   ├── bike/
│   │   ├── bike_001.jpg
│   │   ├── bike_002.jpg
│   │   └── ...
│   └── car/
│       ├── car_001.jpg
│       ├── car_002.jpg
│       └── ...
├── validation/
│   ├── bike/
│   └── car/
└── test/
    ├── bike/
    └── car/
```

### Create Directory Structure

```bash
cd python
python dataset.py
```

This creates the folder structure. Now add your images to each folder.

### Recommended Dataset Sizes

| Split | Bike Images | Car Images |
|-------|------------|-----------|
| Training | 500-2000+ | 500-2000+ |
| Validation | 100-400+ | 100-400+ |
| Test | 100-400+ | 100-400+ |

### Download Sample Dataset

**Kaggle Dataset** (Recommended):
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d utkarshsaxenadn/car-vs-bike-classification-dataset

# Extract to data/ folder
unzip car-vs-bike-classification-dataset.zip -d data/
```

Or collect your own images from Google Images, ensuring diversity in:
- Lighting conditions
- Angles and perspectives
- Vehicle types and colors
- Backgrounds

## 🏋️ Training

### Basic Training

```bash
cd python
python train.py
```

### Training with Custom Parameters

```bash
python train.py \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --img_size 224 \
  --data_dir data
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size (reduce if OOM) |
| `--learning_rate` | 0.001 | Initial learning rate |
| `--img_size` | 224 | Image dimensions (224x224) |
| `--data_dir` | data | Data directory path |

### Monitor Training with TensorBoard

```bash
# In a separate terminal
tensorboard --logdir logs/
```

Open http://localhost:6006 to view:
- Training/validation loss and accuracy
- Precision and recall metrics
- Model graph
- Learning rate schedule

### Training Output

Models are saved to `models/`:
- `best_model.keras`: Best model based on validation accuracy
- `final_model.keras`: Model after all epochs
- `checkpoint_epoch_XX.keras`: Per-epoch checkpoints

Logs are saved to `logs/`:
- `training_log_YYYYMMDD_HHMMSS.csv`: Metrics per epoch
- `training_history.png`: Training curves visualization
- `test_results.json`: Final evaluation metrics
- `training_config.json`: Configuration used

## 🔮 Inference

### Start the API Server

```bash
cd python
python api.py
```

API runs on `http://localhost:5000`

### Start the React Frontend

```bash
# From project root
npm run dev
# or
bun run dev
```

Frontend runs on `http://localhost:5173`

### Using the Web Interface

1. Open http://localhost:5173
2. Upload or drag-and-drop an image
3. Click "Predict" button
4. View classification results with confidence scores

## 📚 API Documentation

### Endpoints

#### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Single Image Prediction

```bash
POST /api/predict
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**
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

{
  "images": ["base64_image1", "base64_image2", ...]
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "prediction": "Bike",
      "confidence": 0.95,
      "probabilities": {"Bike": 0.95, "Car": 0.05}
    },
    {
      "index": 1,
      "prediction": "Car",
      "confidence": 0.92,
      "probabilities": {"Bike": 0.08, "Car": 0.92}
    }
  ]
}
```

#### 4. Model Information

```bash
GET /api/model/info
```

**Response:**
```json
{
  "architecture": "Custom CNN",
  "input_shape": [224, 224, 3],
  "num_classes": 2,
  "class_names": ["Bike", "Car"],
  "trainable_parameters": 7823874,
  "summary": "..."
}
```

#### 5. Training Metrics

```bash
GET /api/model/metrics
```

**Response:**
```json
{
  "test_loss": 0.1234,
  "test_accuracy": 0.9567,
  "test_precision": 0.9523,
  "test_recall": 0.9601
}
```

## 🚢 Deployment

### Option 1: Docker Deployment

```dockerfile
# Dockerfile for Python API
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Copy requirements and install
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python files and models
COPY python/ .
COPY models/ models/

EXPOSE 5000

CMD ["python", "api.py"]
```

Build and run:
```bash
docker build -t bike-car-api .
docker run -p 5000:5000 bike-car-api
```

### Option 2: TensorFlow Serving

Export model:
```python
from model import BikeCarCNN

model = BikeCarCNN()
model.load_model('models/best_model.keras')
model.model.export('exported_model/1/')
```

Serve with Docker:
```bash
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/exported_model,target=/models/bike_car_cnn \
  -e MODEL_NAME=bike_car_cnn \
  tensorflow/serving
```

### Option 3: Cloud Deployment

**AWS SageMaker**:
- Package model with inference script
- Create SageMaker endpoint
- Deploy with auto-scaling

**Google Cloud AI Platform**:
- Upload model to Cloud Storage
- Deploy to AI Platform Prediction
- Configure serving parameters

**Azure ML**:
- Register model in workspace
- Deploy to Azure Container Instances
- Configure endpoint

### Frontend Deployment

Build for production:
```bash
npm run build
# or
bun run build
```

Deploy to:
- **Vercel**: `vercel deploy`
- **Netlify**: `netlify deploy --prod`
- **AWS S3 + CloudFront**: Upload `dist/` folder
- **GitHub Pages**: Push `dist/` to gh-pages branch

## 📁 Project Structure

```
bike-car-classifier/
├── python/                    # Python backend
│   ├── model.py              # CNN model architecture
│   ├── dataset.py            # Dataset handling
│   ├── train.py              # Training script
│   ├── api.py                # Flask API server
│   ├── requirements.txt      # Python dependencies
│   ├── config.json           # Configuration
│   └── README.md             # Python documentation
├── src/                      # React frontend
│   ├── App.tsx               # Main application
│   ├── components/ui/        # UI components
│   ├── index.css             # Styles
│   └── main.tsx              # Entry point
├── data/                     # Dataset (not in repo)
│   ├── train/
│   ├── validation/
│   └── test/
├── models/                   # Trained models (not in repo)
│   ├── best_model.keras
│   └── final_model.keras
├── logs/                     # Training logs
│   ├── training_log_*.csv
│   ├── training_history.png
│   └── tensorboard_*/
├── package.json              # Node dependencies
├── tailwind.config.cjs       # Tailwind config
├── vite.config.ts            # Vite config
└── README.md                 # This file
```

## ⚡ Performance Tips

### Training Speed

**GPU Optimization**:
```python
# Enable mixed precision (2-3x speedup)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

**Increase Batch Size** (if memory allows):
```bash
python train.py --batch_size 64
```

**Use Data Prefetching**:
```python
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
```

### Model Accuracy

1. **More Data**: 1000+ images per class
2. **Data Augmentation**: Already enabled by default
3. **Longer Training**: `--epochs 100`
4. **Learning Rate Tuning**: Try `--learning_rate 0.0001`
5. **Transfer Learning**: Use pre-trained models (ResNet, EfficientNet)

### Inference Speed

1. **TensorFlow Lite**: Convert for mobile/edge
2. **ONNX**: Convert for cross-platform optimization
3. **TensorRT**: NVIDIA GPU acceleration
4. **Batch Predictions**: Use `/api/batch-predict` for multiple images

## 🐛 Troubleshooting

### GPU Not Detected

**Check CUDA installation**:
```bash
nvidia-smi
nvcc --version
```

**Install correct TensorFlow version**:
```bash
pip install tensorflow[and-cuda]
```

**Verify GPU in Python**:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### Out of Memory (OOM)

**Reduce batch size**:
```bash
python train.py --batch_size 16
```

**Reduce image size**:
```bash
python train.py --img_size 128
```

**Enable memory growth**:
```python
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Model Not Found Error

Ensure you've trained the model first:
```bash
cd python
python train.py
```

The API looks for `models/best_model.keras`

### CORS Error in Frontend

Make sure Flask API has CORS enabled (already configured in `api.py`):
```python
from flask_cors import CORS
CORS(app)
```

### Low Accuracy

1. **Check data quality**: Remove corrupted/mislabeled images
2. **Balance dataset**: Equal Bike/Car images
3. **More data**: 1000+ images per class
4. **Longer training**: 100+ epochs
5. **Check data augmentation**: Verify in `dataset.py`

## 📊 Expected Performance

With a well-prepared dataset (1000+ images per class):

| Metric | Expected Range |
|--------|---------------|
| Training Accuracy | 95-99% |
| Validation Accuracy | 92-97% |
| Test Accuracy | 90-96% |
| Inference Time (GPU) | 10-30ms per image |
| Inference Time (CPU) | 50-200ms per image |

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more vehicle types (motorcycle, truck, bus)
- [ ] Implement transfer learning with pre-trained models
- [ ] Add model explainability (GradCAM)
- [ ] Create mobile app (React Native + TFLite)
- [ ] Add real-time video classification
- [ ] Implement model quantization for edge devices

## 📄 License

This project is ready for deployment and production use.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Kaggle community for datasets
- React and Tailwind CSS teams for frontend tools

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the project maintainer.

---

**Built with ❤️ for Computer Vision and Deep Learning**
