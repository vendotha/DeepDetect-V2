import tensorflow as tf
import os

# Define the model paths
keras_model_path = 'model/mesonet_model.h5'
tflite_model_path = 'model/mesonet_model.tflite'

print(f"Loading Keras model from: {keras_model_path}")

# Load the Keras model (we must load it without the optimizer)
model = tf.keras.models.load_model(keras_model_path, compile=False)

print("Keras model loaded successfully.")

# Create a TFLiteConverter object from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply default optimizations (like quantization) to make the file smaller
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model to TensorFlow Lite...")

# Convert the model
tflite_model = converter.convert()

# Ensure the 'model' directory exists
os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)

# Save the converted model to a .tflite file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Successfully converted and saved model to: {tflite_model_path}")
print(f"Original size (HDF5): {os.path.getsize(keras_model_path) / (1024 * 1024):.2f} MB")
print(f"New size (TFLite): {os.path.getsize(tflite_model_path) / (1024 * 1024):.2f} MB")
