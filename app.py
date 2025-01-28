import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import os
import ssl
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
model = tf.keras.models.load_model('./model/model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    img = load_img(file_path, target_size=(64, 64))
    img_array = img_to_array(img)  # Normalize the image
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg, bmp'}), 400

    # Save the file to the uploads folder
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)  # Ensure consistent directory name
    file.save(file_path)

    try:
        # Preprocess the image and make a prediction
        img = preprocess_image(file_path)
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        print('Predicted class is:', predicted_class)

        class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        predicted_label = class_names[predicted_class]

        return jsonify({
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': float(np.max(predictions[0]))
        })
    except Exception as e:
      print(f"Error during prediction: {e}")  # Log the error
      return jsonify({'error': 'Internal server error during prediction'}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
