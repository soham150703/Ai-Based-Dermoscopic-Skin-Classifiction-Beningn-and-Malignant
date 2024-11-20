from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

app = Flask(__name__)
CORS(app)

# ======================= Model 1 (TensorFlow Lite) =======================

# Load the first TensorFlow Lite model
interpreter1 = tf.lite.Interpreter(model_path="C:/Users/Soham Surdas/Desktop/skin/Ai-Based-Dermoscopic-Skin-Classifiction-Beningn-and-Malignant/flask_backend/Skin_Cancer_Model.tflite")
interpreter1.allocate_tensors()

# Helper function to preprocess image for TensorFlow Lite
def preprocess_image_tflite(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict_tflite():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess_image_tflite(img)

        # Prediction using the TensorFlow Lite model
        input_details = interpreter1.get_input_details()
        output_details = interpreter1.get_output_details()
        interpreter1.set_tensor(input_details[0]['index'], img)
        interpreter1.invoke()

        prediction = interpreter1.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction[0])

        result = "Benign" if predicted_class == 0 else "Malignant"

        print(f"Model 1 Prediction: {result}")
        return jsonify({"prediction": result})
    except Exception as e:
        print(f"Error in Model 1: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

# ======================= Model 2 (PyTorch) =======================

# Define GeM pooling class
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        print(f"Initialized GeM with p={self.p.item()} and eps={self.eps}")

    def forward(self, x):
        print(f"Forward pass through GeM with input shape: {x.shape}")
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        print(f"Applying GeM pooling with p={p} and eps={eps}")
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

# Define the ISICModel
class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False):
        super(ISICModel, self).__init__()
        print(f"Creating ISICModel with model_name={model_name}, num_classes={num_classes}, pretrained={pretrained}")
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Checking input features before modifying classifier
        in_features = self.model.classifier.in_features
        print(f"Model input features: {in_features}")
        
        self.model.classifier = nn.Identity()  # Removing the original classifier layer
        self.model.global_pool = nn.Identity()  # Removing the global pooling layer
        self.pooling = GeM()  # Adding GeM pooling
        self.linear = nn.Linear(in_features, num_classes)  # Adding a new linear layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, images):
        print(f"Forward pass through ISICModel with input shape: {images.shape}")
        features = self.model(images)  # Get features from the backbone model
        print(f"Extracted features shape: {features.shape}")
        
        pooled_features = self.pooling(features).flatten(1)  # Apply GeM pooling and flatten
        print(f"Pooled features shape: {pooled_features.shape}")
        
        output = self.sigmoid(self.linear(pooled_features))  # Pass through linear and sigmoid layers
        print(f"Model output shape: {output.shape}")
        
        return output

# Load the PyTorch model
print("Loading model...")
model2 = ISICModel(model_name="tf_efficientnet_b0_ns", num_classes=1)
try:
    model2.load_state_dict(torch.load("best_model.pth", map_location="cpu", weights_only=True), strict=True)

    print("Model weights loaded successfully")
except Exception as e:
    print(f"Error loading model weights: {e}")
model2.eval()

# Preprocessing pipeline for PyTorch model
transform2 = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize the image to match model input size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])

@app.route('/predict2', methods=['POST'])
def predict_torch():
    print("Received request for prediction")
    if 'image' not in request.files:
        print("No image file found in the request")
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        print("Image file found, opening and processing...")
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Debugging image processing
        print(f"Image size before transform: {img.size}")
        
        input_tensor = transform2(img).unsqueeze(0)
        print(f"Input tensor shape after transformation: {input_tensor.shape}")

        # Prediction using the PyTorch model
        with torch.no_grad():
            print("Making prediction with the model...")
            prediction = model2(input_tensor).item()
        
        print(f"Model 2 Prediction: {prediction}")
        return jsonify({"prediction": prediction})
    except Exception as e:
        print(f"Error in Model 2 prediction: {e}")
        return jsonify({"error": str(e)}), 500

# ======================= Run Flask App =======================
if __name__ == "__main__":
    app.run(debug=True)
