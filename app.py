import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import gradio as gr


# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "oral_cancer_model_finetuned.keras")

model = load_model(model_path)
print("[INFO] Model loaded successfully!")


# Grad-CAM Layer
last_conv_layer_name = "Conv_1"

base_model = Model(
    inputs=model.input,
    outputs=model.get_layer(last_conv_layer_name).output
)

print("[INFO] Using Grad-CAM layer:", last_conv_layer_name)


# Grad-CAM Function
def make_gradcam_heatmap(img_array, base_model, model, last_conv_layer_name):

    grad_model = Model(
        inputs=model.inputs,
        outputs=[base_model.output, model.output]
    )

    with tf.GradientTape() as tape:

        last_conv_output, preds = grad_model(img_array)

        pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_output = last_conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        last_conv_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_output, axis=-1)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


# Prediction Function
def predict_and_explain(img):

    img = img.resize((224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)

    prob = preds[0][0]

    percent = round(prob * 100, 2)

    label = "Malignant (Cancer)" if prob >= 0.5 else "Benign (Non-Cancer)"

    result_text = f"Prediction: {label}\nProbability: {percent}%"

    heatmap = make_gradcam_heatmap(
        img_array,
        base_model,
        model,
        last_conv_layer_name
    )

    original_img = np.array(img)

    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        original_img_bgr,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    superimposed_img = cv2.cvtColor(
        superimposed_img,
        cv2.COLOR_BGR2RGB
    )

    return result_text, superimposed_img


# Gradio Interface
iface = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Image(type="numpy", label="Grad-CAM Heatmap")
    ],
    title="Oral Cancer Detection with Grad-CAM",
    description="Upload an oral lesion image to detect possible cancer presence and view Grad-CAM explanation."
)


# Run Application
if __name__ == "__main__":
    iface.launch()