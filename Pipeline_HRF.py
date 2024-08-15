from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
import tensorflow as tf
from tensorflow.keras.models import Model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numba import jit, cuda 

# Check for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
 
### MODEL ###
def create_unet(input_shape=(None, None, 3)):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)  

    up1 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2), conv1], axis=3)
    conv3 = Conv2D(32, 3, activation='relu', padding='same')(up1)
    conv3 = Conv2D(32, 3, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(32, 3, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(32, 3, activation='relu', padding='same')(conv3)  
    outputs = Conv2D(1, 1, activation='sigmoid')(conv3)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

unet_model = create_unet()

### DATA ###
images_path = 'HRF-database/images/'
segmented_images_path = 'HRF-database/manual1/'

original_images = []
segmentation_images = []

    # 25% of original
target_height = 584
target_width = 876

for filename in os.listdir(images_path):
    if filename.endswith('.JPG') or filename.endswith('.jpg'):
        original_image = cv2.imread(os.path.join(images_path, filename))
        if original_image is not None:
            resized_image = cv2.resize(original_image, (target_width, target_height))
            original_images.append(resized_image)
        else:
            print(f"Error loading image: {filename}")

for filename in os.listdir(segmented_images_path):
    if filename.endswith('.tif'):
        segmented_image_path = os.path.join(segmented_images_path, filename)
        segmented_image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
        if segmented_image is not None:
            resized_segmented_image = cv2.resize(segmented_image, (target_width, target_height))
            segmentation_images.append(resized_segmented_image)
        else:
            print(f"Error loading segmentation image: {filename}")

X_train, X_val, y_train, y_val = train_test_split(original_images, segmentation_images, test_size=0.2, random_state=42)

X_train = np.array(X_train) / 255
X_val = np.array(X_val) / 255

y_train = np.expand_dims(y_train, axis=-1) / 255
y_val = np.expand_dims(y_val, axis=-1) / 255

### TRAINING ***
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
@jit(target_backend=cuda) 
def trainCNN():
     unet_model.fit(X_train, y_train, batch_size=6, epochs=35) #callbacks=[tensorboard_callback])
       
trainCNN()

### EVALUUATION ###
y_pred = unet_model.predict(X_val)

def apply_threshold(pred, threshold=0.42):
  pred_bin = np.where(pred >= threshold, 1, 0)

  return pred_bin

def compare_pred_mask(pred, mask):
  pred_bin = pred
  mask_bin = mask

  tp = np.sum(pred_bin * mask_bin)
  tn = np.sum((1 - pred_bin) * (1 - mask_bin))
  fp = np.sum(pred_bin * (1 - mask_bin))
  fn = np.sum((1 - pred_bin) * mask_bin)

  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1_score = 2 * (precision * recall) / (precision + recall)

  return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}

### METRICS ###
metrics_per_image = []
for i in range(len(X_val)):
  pred = y_pred[i][:, :, 0]
  mask = y_val[i][:, :, 0]

  pred_bin = apply_threshold(pred)
  metrics_per_image.append(compare_pred_mask(pred_bin, mask))

metrics_avg = {metric: np.mean([m[metric] for m in metrics_per_image]) for metric in metrics_per_image[0]}

### OUTPUT ###
with open("metrics.txt", 'w') as outfile:
    outfile.write("Model metrics:\n")
    for metric, value in metrics_avg.items():
        outfile.write(f"{metric}: {value:.4f}\n")
  
pred_1 = y_pred[0][:, :, 0]
mask_1 = y_val[0][:, :, 0]
plt.figure(figsize=(16, 12))

plt.subplot(1, 2, 1)
plt.imshow(pred_1, cmap='gray')
plt.title('Vessels segmenation with UNet')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask_1, cmap='gray')
plt.title('Segmented Image - goal')
plt.axis('off')

plt.savefig('segmentation_comparison.png')