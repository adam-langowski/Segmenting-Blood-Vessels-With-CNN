from flask import Flask, request, render_template, send_file
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import io

app = Flask(__name__)
model = tf.keras.models.load_model('model_segmentingVessels.keras')

def segment_image(image, model):
    target_height = 584
    target_width = 876
    resized_image = cv2.resize(image, (target_width, target_height))
    resized_image = np.expand_dims(resized_image, axis=0) / 255.0
    prediction = model.predict(resized_image)
    return prediction[0]

def apply_threshold(pred, threshold=0.42):
    binary_image = np.where(pred >= threshold, 1, 0)
    return binary_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)
    
    prediction = segment_image(image, model)
    segmented_image = apply_threshold(prediction)
    
    segmented_image = (segmented_image * 255).astype(np.uint8)
    segmented_image_pil = Image.fromarray(segmented_image[:,:,0], mode='L')
    
    byte_io = io.BytesIO()
    segmented_image_pil.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    return send_file(byte_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
