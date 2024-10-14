from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="D:/edai/Skin_Cancer_Model.tflite")
interpreter.allocate_tensors()

# Helper function to preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to the input shape of the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess_image(img)

        # Make prediction using the TensorFlow Lite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        # Get the model's prediction
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction[0])

        if predicted_class == 0:
            result = "Benign"
        else:
            result = "Malignant"

        print(f"Prediction: {result}")  # Print to terminal for debugging

        return jsonify({"prediction": result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == "__main__":
    app.run(debug=True)
