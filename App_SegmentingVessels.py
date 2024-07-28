from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import cv2
import io
import base64
import time
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Check for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("No GPU found, using CPU")

models = {
    "HRF": tf.keras.models.load_model('model_segmentingVessels.keras'),
    "DRIVE": tf.keras.models.load_model('model_DRIVE_segmentingVessels.keras')
}

def segment_image(image, model):
    target_height = 584
    target_width = 876
    start_time = time.time()
    resized_image = cv2.resize(image, (target_width, target_height))
    print(f"Resizing time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    resized_image = np.expand_dims(resized_image, axis=0) / 255.0
    prediction = model.predict(resized_image)
    print(f"Model prediction time: {time.time() - start_time:.2f} seconds")

    return prediction[0]

def apply_threshold(pred, threshold=0.42):
    binary_image = np.where(pred >= threshold, 1, 0)
    return binary_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    start_time = time.time()
    model_choice = request.form['model']
    model = models.get(model_choice, models['HRF'])
    
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print(f"Image loading time: {time.time() - start_time:.2f} seconds")

    prediction = segment_image(image, model)
    segmented_image = apply_threshold(prediction)
    
    segmented_image = (segmented_image * 255).astype(np.uint8)
    
    start_time = time.time()
    _, buffer = cv2.imencode(".png", segmented_image)
    print(f"Image encoding time: {time.time() - start_time:.2f} seconds")

    io_buf = io.BytesIO(buffer)
    image_base64 = base64.b64encode(io_buf.getvalue()).decode('utf-8')
    
    return jsonify({'image': image_base64})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)