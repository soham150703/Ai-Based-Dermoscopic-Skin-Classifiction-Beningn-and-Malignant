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

# Load the TensorFlow Lite model
interpreter1 = tf.lite.Interpreter(model_path="D:/Edai/model3.tflite")
interpreter1.allocate_tensors()

# Helper function to preprocess image for TensorFlow Lite
def preprocess_image_tflite(image, target_size=(224, 224)):
    img = image.resize(target_size).convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict_tflite():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = preprocess_image_tflite(img)

        # Debugging: Print input details
        input_details = interpreter1.get_input_details()
        output_details = interpreter1.get_output_details()

        print("Input details:", input_details)
        print("Input image shape:", img.shape)
        print("Input dtype:", img.dtype)

        # Perform prediction
        interpreter1.set_tensor(input_details[0]['index'], img)
        interpreter1.invoke()
        prediction = interpreter1.get_tensor(output_details[0]['index'])

        print("Raw prediction:", prediction)
        predicted_class = np.argmax(prediction[0])
        result = "Benign" if predicted_class == 0 else "Malignant"

        print(f"Model 1 Prediction: {result}")
        return jsonify({"prediction": result})
    except Exception as e:
        print(f"Error in Model 1: {e}")
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# ======================= Model 2 (PyTorch) =======================

# Define GeM pooling class
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

# Define the ISICModel
class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output

# Load the PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = ISICModel(model_name="tf_efficientnet_b0_ns", num_classes=1).to(device)

try:
    model2.load_state_dict(torch.load("best_model.pth", map_location=device), strict=True)
    print("PyTorch model loaded successfully")
except Exception as e:
    print(f"Error loading PyTorch model weights: {e}")

model2.eval()

# Preprocessing pipeline for PyTorch model
transform2 = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict2', methods=['POST'])
def predict_torch():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform2(img).unsqueeze(0).to(device)

        # Parse the custom threshold from request (default is 0.5)
        threshold = float(request.form.get('threshold', 0.5))

        # Prediction using the PyTorch model
        with torch.no_grad():
            prediction = model2(input_tensor).item()

        classification = "Malignant" if prediction >= threshold else "Benign"

        # Return prediction score and classification
        return jsonify({
            "prediction_score": round(float(prediction), 6),
            "classification": classification,
            "threshold": threshold
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"An error occurred in PyTorch model: {e}"}), 500

# ======================= Run Flask App =======================
if __name__ == "__main__":
    import torch

    # Ensure the device is set properly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run the app
    app.run(debug=True, host="0.0.0.0", port=8000)