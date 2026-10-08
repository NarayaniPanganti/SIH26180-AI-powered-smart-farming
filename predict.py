"""Classify one leaf image.

    python predict.py sample_images/leaf1.jpg
    python predict.py leaf.jpg --threshold 0.7
Uses leaf_model.tflite if present, otherwise leaf_model.h5.
Class names come from classes.txt (written by train_model.py).
"""
import argparse, os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("--classes", default="classes.txt")
p.add_argument("--threshold", type=float, default=0.6,
               help="below this confidence the bot should NOT spray; it flags 'uncertain'")
args = p.parse_args()

classes = open(args.classes).read().split("\n")
img = Image.open(args.image).convert("RGB").resize((224, 224))
x = preprocess_input(np.expand_dims(np.array(img, dtype=np.float32), 0))

if os.path.exists("leaf_model.tflite"):
    interp = tf.lite.Interpreter(model_path="leaf_model.tflite")
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    probs = interp.get_tensor(out["index"])[0]
else:
    probs = tf.keras.models.load_model("leaf_model.h5").predict(x, verbose=0)[0]

order = np.argsort(probs)[::-1]
for i in order[:3]:
    print(f"{classes[i]:30s} {probs[i]*100:5.1f}%")
best = order[0]
if probs[best] < args.threshold:
    print("-> UNCERTAIN: no spraying, log position for farmer review")
else:
    print(f"-> DETECTED: {classes[best]} (confidence {probs[best]*100:.1f}%)")
