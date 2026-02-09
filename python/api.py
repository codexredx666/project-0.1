"""
Flask API for Bike vs Car Classification Model
Provides inference endpoints for the trained CNN model
"""

import os
import io
import json
import numpy as np
from pathlib import Path
from PIL import Image
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS

from model import BikeCarCNN
from dataset import BikeCarDataset


app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model instance
model = None
dataset_handler = None
CLASS_NAMES = ['Bike', 'Car']


def initialize_model(model_path='models/best_model.keras'):
    """Initialize the model on startup"""
    global model, dataset_handler
    
    try:
        model = BikeCarCNN()
        
        # Try to load model
        if Path(model_path).exists():
            model.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠ Model not found at {model_path}")
            print("  Please train the model first using train.py")
            model = None
        
        # Initialize dataset handler for preprocessing
        dataset_handler = BikeCarDataset()
        print("✓ Dataset handler initialized")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        model = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict bike or car from uploaded image
    
    Accepts:
    - multipart/form-data with 'image' file
    - JSON with base64 encoded image
    
    Returns:
    - prediction: class name (Bike/Car)
    - confidence: prediction confidence (0-1)
    - probabilities: array of probabilities for each class
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 503
    
    try:
        # Get image from request
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            image = Image.open(file.stream)
        elif request.json and 'image' in request.json:
            # Base64 encoded image
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        processed_image = dataset_handler.preprocess_image(image)
        
        # Make prediction
        class_name, confidence = model.predict_class(
            processed_image, 
            class_names=CLASS_NAMES
        )
        
        # Get all probabilities
        probabilities = model.predict(processed_image)[0]
        
        response = {
            'prediction': class_name,
            'confidence': float(confidence),
            'probabilities': {
                CLASS_NAMES[i]: float(probabilities[i]) 
                for i in range(len(CLASS_NAMES))
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        # Get model summary as string
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = summary_buffer = StringIO()
        model.model.summary()
        sys.stdout = old_stdout
        summary = summary_buffer.getvalue()
        
        # Get trainable parameters
        trainable_params = model.model.count_params()
        
        info = {
            'architecture': 'Custom CNN',
            'input_shape': model.input_shape,
            'num_classes': model.num_classes,
            'class_names': CLASS_NAMES,
            'trainable_parameters': trainable_params,
            'summary': summary
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/metrics', methods=['GET'])
def model_metrics():
    """Get model training metrics if available"""
    metrics_file = Path('logs/test_results.json')
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics), 200
    else:
        return jsonify({
            'message': 'No metrics available. Train the model to generate metrics.'
        }), 404


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict multiple images at once
    
    Accepts:
    - JSON with array of base64 encoded images
    
    Returns:
    - Array of predictions
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        data = request.json
        
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({'error': 'Expected array of images'}), 400
        
        results = []
        
        for idx, image_data in enumerate(data['images']):
            try:
                # Decode base64 image
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Preprocess and predict
                processed_image = dataset_handler.preprocess_image(image)
                class_name, confidence = model.predict_class(
                    processed_image, 
                    class_names=CLASS_NAMES
                )
                
                probabilities = model.predict(processed_image)[0]
                
                results.append({
                    'index': idx,
                    'prediction': class_name,
                    'confidence': float(confidence),
                    'probabilities': {
                        CLASS_NAMES[i]: float(probabilities[i]) 
                        for i in range(len(CLASS_NAMES))
                    }
                })
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e)
                })
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Run Flask server
    print("\n" + "="*60)
    print("Starting Bike vs Car Classification API")
    print("="*60)
    print("Server running on http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /api/predict         - Single image prediction")
    print("  POST /api/batch-predict   - Multiple images prediction")
    print("  GET  /api/model/info      - Model information")
    print("  GET  /api/model/metrics   - Training metrics")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
