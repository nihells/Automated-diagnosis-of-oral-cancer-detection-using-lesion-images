import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
import os
import warnings
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

if not os.path.exists('models'):
    os.makedirs('models')

train_dir = 'data/'
test_dir = 'test_data/'  # Update as needed

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print("Training, validation, and test datasets loaded successfully!")

inputs = Input(shape=(224, 224, 3))
base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('models/best_oral_cancer_model.keras',
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=1)

initial_epochs = 5
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=initial_epochs,
    callbacks=[early_stopping, model_checkpoint]
)

print("Initial training completed.")
val_loss, val_acc = model.evaluate(val_generator)
print(f"Initial Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Initial Validation Loss: {val_loss:.4f}")

# Updated Keras format
model.save('models/oral_cancer_model_initial.keras')
print("Initial model saved.")

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stopping, model_checkpoint]
)

print("Fine-tuning completed.")
val_loss, val_acc = model.evaluate(val_generator)
print(f"Fine-tuned Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Fine-tuned Validation Loss: {val_loss:.4f}")

# Updated Keras format
model.save('models/oral_cancer_model_finetuned.keras')
print("Fine-tuned model saved.")

print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

test_generator.reset()
preds_prob = model.predict(test_generator)
preds = (preds_prob > 0.5).astype(int).flatten()
true_labels = test_generator.classes

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=list(test_generator.class_indices.keys())))
print("Confusion Matrix:")
print(confusion_matrix(true_labels, preds))

def plot_training_history(hist, title):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history, "Initial Training")
plot_training_history(history_fine, "Fine-tuning")

def make_gradcam_heatmap(img_array, base_model, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[base_model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8  # Prevent division by zero
    return heatmap

def display_gradcam(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)
    print(f"Prediction (malignant probability): {preds[0][0]:.4f}")
    heatmap = make_gradcam_heatmap(img_array, base_model, model, last_conv_layer_name='out_relu')
    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + orig_img
    plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Run Grad-CAM on example image
sample_image = 'data/benign/001.jpeg'  # Adjust path/filename accordingly
display_gradcam(sample_image)
