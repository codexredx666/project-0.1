# 🚢 Deployment Guide

Complete guide for deploying the Bike vs Car CNN Classifier to production.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Model Export Formats](#model-export-formats)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Platforms](#cloud-platforms)
5. [Frontend Deployment](#frontend-deployment)
6. [Production Optimization](#production-optimization)
7. [Monitoring & Maintenance](#monitoring--maintenance)

## Pre-Deployment Checklist

Before deploying to production:

- [ ] Model trained and validated (>90% accuracy)
- [ ] Test set evaluation completed
- [ ] Model saved in multiple formats (Keras, TFLite, ONNX)
- [ ] API endpoints tested locally
- [ ] Frontend built and optimized
- [ ] Environment variables configured
- [ ] CORS settings configured
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Rate limiting implemented (if needed)
- [ ] Security headers added
- [ ] SSL/HTTPS certificates ready

## Model Export Formats

### 1. Keras Format (Default)

Already saved during training:
```python
# models/best_model.keras
```

### 2. TensorFlow SavedModel

For TensorFlow Serving:

```python
from model import BikeCarCNN

# Load trained model
model = BikeCarCNN()
model.load_model('models/best_model.keras')

# Export
model.model.export('exported_model/1/')
```

Directory structure:
```
exported_model/
└── 1/
    ├── saved_model.pb
    ├── variables/
    │   ├── variables.data-00000-of-00001
    │   └── variables.index
    └── assets/
```

### 3. TensorFlow Lite (Mobile/Edge)

For mobile apps and edge devices:

```python
import tensorflow as tf
from model import BikeCarCNN

# Load model
model = BikeCarCNN()
model.load_model('models/best_model.keras')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model.model)

# Optional: Optimize for size and speed
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 4. ONNX Format (Cross-Platform)

For deployment on non-TensorFlow platforms:

```bash
# Install converter
pip install tf2onnx

# Convert
python -m tf2onnx.convert \
  --keras models/best_model.keras \
  --output model.onnx \
  --opset 13
```

## Docker Deployment

### Full Stack Deployment

**Dockerfile for Python API**:

```dockerfile
# Dockerfile.api
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY python/ .

# Create necessary directories
RUN mkdir -p models logs

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run API
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "api:app"]
```

**Dockerfile for React Frontend**:

```dockerfile
# Dockerfile.frontend
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Build
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - api
    restart: unless-stopped
```

**Deploy with Docker Compose**:

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Cloud Platforms

### AWS Deployment

#### Option 1: AWS ECS (Elastic Container Service)

1. **Push Docker images to ECR**:
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push API
docker tag bike-car-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/bike-car-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/bike-car-api:latest

# Tag and push frontend
docker tag bike-car-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/bike-car-frontend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/bike-car-frontend:latest
```

2. **Create ECS task definition** with both containers

3. **Deploy to ECS cluster** with load balancer

#### Option 2: AWS SageMaker

1. **Package model**:
```python
# sagemaker_deploy.py
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

# Package model
model = TensorFlowModel(
    model_data='s3://my-bucket/models/model.tar.gz',
    role='SageMakerRole',
    framework_version='2.15',
    py_version='py310'
)

# Deploy
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

2. **Create inference script**:
```python
# inference.py
import tensorflow as tf
import json
import numpy as np

def model_fn(model_dir):
    model = tf.keras.models.load_model(f'{model_dir}/1')
    return model

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['instances'])

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    return json.dumps({'predictions': prediction.tolist()})
```

### Google Cloud Platform

#### Option 1: Cloud Run

1. **Build and push to GCR**:
```bash
# Build
gcloud builds submit --tag gcr.io/PROJECT_ID/bike-car-api

# Deploy
gcloud run deploy bike-car-api \
  --image gcr.io/PROJECT_ID/bike-car-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Option 2: AI Platform

1. **Upload model to Cloud Storage**:
```bash
gsutil cp -r exported_model/ gs://my-bucket/models/bike_car_cnn/
```

2. **Create model version**:
```bash
gcloud ai-platform models create bike_car_cnn
gcloud ai-platform versions create v1 \
  --model bike_car_cnn \
  --origin gs://my-bucket/models/bike_car_cnn/1/ \
  --runtime-version 2.15 \
  --framework tensorflow \
  --python-version 3.10
```

3. **Make predictions**:
```python
import googleapiclient.discovery

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/PROJECT_ID/models/bike_car_cnn/versions/v1'

response = service.projects().predict(
    name=name,
    body={'instances': [preprocessed_image.tolist()]}
).execute()
```

### Azure ML

1. **Register model**:
```python
from azureml.core import Workspace, Model

ws = Workspace.from_config()
model = Model.register(
    workspace=ws,
    model_path='models/best_model.keras',
    model_name='bike-car-cnn'
)
```

2. **Create scoring script**:
```python
# score.py
import json
import tensorflow as tf
import numpy as np

def init():
    global model
    model_path = Model.get_model_path('bike-car-cnn')
    model = tf.keras.models.load_model(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    predictions = model.predict(np.array(data))
    return json.dumps(predictions.tolist())
```

3. **Deploy**:
```python
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)

aci_config = AciWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=4
)

service = Model.deploy(
    workspace=ws,
    name='bike-car-classifier',
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)
```

## Frontend Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel deploy --prod
```

### Netlify

```bash
# Install Netlify CLI
npm i -g netlify-cli

# Build
npm run build

# Deploy
netlify deploy --prod --dir=dist
```

### AWS S3 + CloudFront

```bash
# Build
npm run build

# Upload to S3
aws s3 sync dist/ s3://my-bucket --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id DIST_ID --paths "/*"
```

**nginx.conf** for custom server:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API proxy (if needed)
    location /api {
        proxy_pass http://api:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # React Router (SPA)
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

## Production Optimization

### API Optimization

1. **Use Gunicorn with multiple workers**:
```bash
gunicorn --bind 0.0.0.0:5000 \
  --workers 4 \
  --threads 2 \
  --timeout 120 \
  --worker-class gthread \
  api:app
```

2. **Enable caching**:
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/predict', methods=['POST'])
@cache.cached(timeout=300, key_prefix='predict')
def predict():
    # ...
```

3. **Add rate limiting**:
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/predict')
@limiter.limit("10 per minute")
def predict():
    # ...
```

### Model Optimization

1. **Quantization (reduce size by 4x)**:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

2. **Pruning (reduce size by 2-3x)**:
```python
import tensorflow_model_optimization as tfmot

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model.model, **pruning_params
)
```

## Monitoring & Maintenance

### Logging Setup

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler('logs/api.log', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Log predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    app.logger.info(f'Prediction request from {request.remote_addr}')
    # ...
    app.logger.info(f'Prediction result: {result["prediction"]} ({result["confidence"]:.2f})')
```

### Health Checks

```python
@app.route('/health')
def health_check():
    checks = {
        'api': 'ok',
        'model_loaded': model is not None,
        'disk_space': get_disk_usage() < 90,
        'memory': get_memory_usage() < 90
    }
    
    status_code = 200 if all(checks.values()) else 503
    return jsonify(checks), status_code
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.route('/api/predict')
@prediction_duration.time()
def predict():
    prediction_counter.inc()
    # ...

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Performance Monitoring

Use New Relic, Datadog, or custom monitoring:

```python
import time

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    app.logger.info(f'{request.method} {request.path} {response.status_code} {duration:.3f}s')
    return response
```

## Security Considerations

1. **Environment Variables**: Never commit secrets
2. **HTTPS**: Always use SSL in production
3. **CORS**: Restrict to specific origins
4. **Rate Limiting**: Prevent abuse
5. **Input Validation**: Validate image size/format
6. **Authentication**: Add API keys if needed
7. **Firewall**: Restrict access to known IPs

## Troubleshooting Production Issues

| Issue | Solution |
|-------|----------|
| Slow predictions | Increase workers, use GPU, enable caching |
| Out of memory | Reduce batch size, limit concurrent requests |
| Model not loading | Check file permissions, paths |
| CORS errors | Update CORS config in api.py |
| High latency | Add CDN, optimize images, use edge locations |

---

**Ready for Production! 🚀**
