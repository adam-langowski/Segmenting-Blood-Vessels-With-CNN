import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import cv2
import threading


def load_image():
    global original_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        original_image = cv2.imread(file_path)
        display_image(original_image)
        return original_image
    return None


def segment_image(image, model_path):
    target_height = 584
    target_width = 876
    model = tf.keras.models.load_model(model_path)

    resized_image = cv2.resize(image, (target_width, target_height))
    resized_image = np.array(resized_image) / 255.0
    resized_image = np.expand_dims(resized_image, axis=0)

    prediction = model.predict(resized_image)
    return prediction


def apply_threshold(pred, threshold=0.42):
    pred_bin = np.where(pred >= threshold, 1, 0)
    return pred_bin.astype(np.uint8)  


def segment_button_click():
    if original_image is None:
        messagebox.showerror("Error", "Please load an image.")
        return

    def segment_in_thread():
        prediction = segment_image(original_image.copy(), "model_segmentingVessels.keras")
        segmented_image = apply_threshold(prediction)
        segmented_image = segmented_image[:, :, 0]
        cv2.imshow("Segmented Image", segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    threading.Thread(target=segment_in_thread).start()


def display_image(image):
    global img_label
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((876, 584))
    photo = ImageTk.PhotoImage(img)
    img_label.config(image=photo)
    img_label.image = photo


root = tk.Tk()
root.title("Retinal Blood Vessels Segmentation App")
root.geometry("900x720")

btn_load_image = tk.Button(root, text="Load Image", command=lambda: display_image(load_image()))
btn_load_image.pack(pady=10)

btn_segment = tk.Button(root, text="Segment", command=segment_button_click)
btn_segment.pack(pady=5)

img_label = tk.Label(root)
img_label.pack()

original_image = None
segmented_image = None

root.mainloop()
