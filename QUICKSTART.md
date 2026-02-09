# ⚡ Quick Start Guide

Get your Bike vs Car CNN classifier running in 5 minutes!

## 🚀 Fast Setup (Development)

### 1. Install Dependencies

```bash
# Python backend
cd python
pip install -r requirements.txt

# React frontend (from project root)
cd ..
npm install
```

### 2. Setup Dataset

```bash
cd python

# Create directory structure
python dataset.py

# Download sample dataset (choose one):

# Option A: Kaggle (requires Kaggle CLI)
kaggle datasets download -d utkarshsaxenadn/car-vs-bike-classification-dataset
unzip car-vs-bike-classification-dataset.zip -d data/

# Option B: Manual
# Place your images in:
#   data/train/bike/
#   data/train/car/
#   data/validation/bike/
#   data/validation/car/
#   data/test/bike/
#   data/test/car/
```

### 3. Train Model

```bash
# Basic training (50 epochs)
python train.py

# Or custom training
python train.py --epochs 100 --batch_size 32
```

**Training time**:
- GPU: 10-30 minutes (depending on dataset size)
- CPU: 2-6 hours

### 4. Start Services

```bash
# Terminal 1: Start API server
cd python
python api.py

# Terminal 2: Start React frontend (from project root)
npm run dev
```

### 5. Open Browser

Navigate to: http://localhost:5173

Upload an image and click "Predict"!

---

## 🎯 Quick Commands Reference

### Python Backend

```bash
cd python

# Check GPU
python model.py

# Create dataset structure
python dataset.py

# Train model
python train.py

# Start API
python api.py

# Monitor training
tensorboard --logdir logs/
```

### React Frontend

```bash
# Development
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 🐛 Common Issues & Quick Fixes

### GPU Not Found

```bash
# Install CUDA version of TensorFlow
pip install tensorflow[and-cuda]

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 16

# Or reduce image size
python train.py --img_size 128
```

### Model Not Found

```bash
# Make sure you trained the model first
cd python
python train.py

# Check if model exists
ls models/best_model.keras
```

### API Connection Error

1. Ensure API is running: `python api.py`
2. Check port 5000 is available
3. Verify CORS is enabled in `api.py`

---

## 📊 Expected Results

After training with 1000+ images per class:

| Metric | Value |
|--------|-------|
| Training Accuracy | 95-99% |
| Validation Accuracy | 92-97% |
| Test Accuracy | 90-96% |
| Inference Time (GPU) | 10-30ms |
| Inference Time (CPU) | 50-200ms |

---

## 🎨 What You Get

✅ **Python Backend**:
- Custom CNN model architecture
- GPU-accelerated training
- Data augmentation
- Model checkpointing
- TensorBoard logging
- Flask REST API

✅ **React Frontend**:
- Beautiful UI with drag-and-drop
- Real-time predictions
- Confidence scores
- Probability visualization
- Responsive design

✅ **Production Ready**:
- Docker support
- Cloud deployment guides
- Model export formats
- Comprehensive documentation

---

## 📚 Next Steps

1. **Improve Model**:
   - Add more training data
   - Experiment with hyperparameters
   - Try transfer learning

2. **Customize Frontend**:
   - Change colors in `src/index.css`
   - Add more features
   - Integrate with your app

3. **Deploy to Production**:
   - Follow `DEPLOYMENT.md`
   - Use Docker containers
   - Deploy to AWS/GCP/Azure

4. **Extend Functionality**:
   - Add more vehicle classes
   - Implement video classification
   - Create mobile app

---

## 🆘 Need Help?

- 📖 Read full documentation: `README.md`
- 🚀 Deployment guide: `DEPLOYMENT.md`
- 🐍 Python docs: `python/README.md`
- 💬 Open an issue on GitHub

---

**Happy Classifying! 🚴🚗**
