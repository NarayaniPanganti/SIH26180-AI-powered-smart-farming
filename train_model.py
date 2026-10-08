"""Train a leaf-condition classifier (MobileNetV2 transfer learning).

Expected folder layout:
    dataset/
        class_a/ img1.jpg ...
        class_b/ ...
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# ---- 0. sanity check the dataset (a flat ~50% curve usually means a data/label problem)
class_dirs = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
print("Images per class:")
for d in class_dirs:
    n = len(os.listdir(os.path.join(DATA_DIR, d)))
    print(f"  {d:30s} {n}")
    if n < 100:
        print(f"  WARNING: '{d}' has fewer than 100 images - results may be unreliable")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.class_names
print("Classes found:", class_names)
with open("classes.txt", "w") as f:          # predict.py reads this
    f.write("\n".join(class_names))

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.1),
])
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("leaf_model.h5")

# ---- training curve
plt.figure()
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
plt.savefig("training_curve.png", dpi=120)

# ---- confusion matrix on the validation set
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_pred = np.argmax(model.predict(val_ds), axis=1)
print(classification_report(y_true, y_pred, target_names=class_names))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=class_names).plot(xticks_rotation=45)
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120)
print("Saved leaf_model.h5, classes.txt, training_curve.png, confusion_matrix.png")
