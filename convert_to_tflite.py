"""Shrink the Keras model for edge devices (Raspberry Pi / Jetson)."""
import tensorflow as tf

model = tf.keras.models.load_model("leaf_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
open("leaf_model.tflite", "wb").write(converter.convert())
print("Saved leaf_model.tflite")
