#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Animal90 CNN Classifier
Author: Mahmud Huseynov
Created: 2025-07-05
Description: 
    This standalone script trains a Convolutional Neural Network (CNN) to recognise
    all 90 animal species in the "Animal Image Dataset – 90 Different Animals" from Kaggle.
    It applies data augmentation, dropout regularisation, early stopping, and batch
    inference for real-world performance.

Usage
-----
1. Download and unzip the Kaggle dataset so that each animal class is a
   sub-folder containing images, e.g.
       ~/datasets/animals90/antelope/xxx.jpg
       ~/datasets/animals90/badger/xxx.jpg
2. Optionally adjust `DATASET_DIR` below to match your path.
3. Run:
       python animal90_cnn.py         # trains and evaluates the model
       python animal90_cnn.py --predict /path/to/test_images

Tested with: Python 3.10+, TensorFlow 2.16+, NVIDIA GPU (optional).
"""

import argparse
import os
import sys
from pathlib import Path
import tensorflow as tf

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
DATASET_DIR = Path(os.getenv('ANIMALS90_DIR', r'C:\Users\HUAWEI\Desktop\Animal Classifier\animals\animals')).expanduser()
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
RANDOM_SEED = 42
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
AUTOTUNE = tf.data.AUTOTUNE

# --------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------

def sanity_check_dataset(root: Path):
    """Ensure the dataset directory exists with subfolders for each class."""
    if not root.exists():
        print(f"[ERROR] Dataset directory '{root}' not found.\n"
              "Download from Kaggle: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals")
        sys.exit(1)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[ERROR] '{root}' contains no subdirectories. Ensure class folders exist.")
        sys.exit(1)
    print(f"✔ Dataset structure looks OK ({len(subdirs)} classes found).")


def build_datasets(root: Path):
    """Return train, val, test tf.data.Dataset objects for ALL classes."""
    # Training split (reserving val+test)
    train_load_ds = tf.keras.preprocessing.image_dataset_from_directory(
        root,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=VAL_SPLIT + TEST_SPLIT,
        subset='training',
        seed=RANDOM_SEED,
    )
    # Grab class names here before augmentation
    class_names = train_load_ds.class_names

    # Validation+Test split
    valtest_ds = tf.keras.preprocessing.image_dataset_from_directory(
        root,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=VAL_SPLIT + TEST_SPLIT,
        subset='validation',
        seed=RANDOM_SEED,
    )

    # Split valtest into val and test
    total = tf.data.experimental.cardinality(valtest_ds).numpy()
    val_count = int((VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)) * total)
    val_ds = valtest_ds.take(val_count).prefetch(AUTOTUNE)
    test_ds = valtest_ds.skip(val_count).prefetch(AUTOTUNE)

    # Data augmentation pipeline for train
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    train_ds = (
        train_load_ds
        .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate(train_ds, val_ds, test_ds, class_names):
    model = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )
    train_acc = history.history['accuracy'][-1] * 100
    print(f"Training accuracy: {train_acc:.2f}%")
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {acc*100:.2f}%")
    model.save('animal90_cnn_savedmodel.keras')
    print("Model saved to ./animal90_cnn_savedmodel.keras")
    return model


def predict_folder(model_path: Path, folder: Path, class_names):
    """Run batch inference on all images in *folder* and print predictions."""
    if not Path(model_path).exists():
        print(f"[ERROR] Saved model '{model_path}' not found.")
        sys.exit(1)
    model = tf.keras.models.load_model(model_path)
    img_files = [p for p in folder.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not img_files:
        print(f"No images found in {folder}")
        return
    print("file,predicted_class,confidence")
    for img in img_files:
        img_raw = tf.keras.utils.load_img(img, target_size=IMG_SIZE)
        x = tf.keras.utils.img_to_array(img_raw)
        x = tf.expand_dims(x, 0)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        preds = model.predict(x, verbose=0)[0]
        idx = tf.argmax(preds).numpy()
        conf = preds[idx]
        print(f"{img.name},{class_names[idx]},{conf:.3f}")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train or use the Animal90 CNN.")
    parser.add_argument('--predict', type=str, default=None,
                        help='Directory of images for batch inference (skip training)')
    args = parser.parse_args()

    print("TensorFlow version:", tf.__version__)
    if tf.config.list_physical_devices('GPU'):
        print("✔ GPU detected – training will be fast.")
    else:
        print("⚠ No GPU detected – expect longer training time.")

    sanity_check_dataset(DATASET_DIR)
    train_ds, val_ds, test_ds, class_names = build_datasets(DATASET_DIR)

    if args.predict:
        predict_folder('animal90_cnn_savedmodel.keras', Path(args.predict), class_names)
    else:
        train_and_evaluate(train_ds, val_ds, test_ds, class_names)


if __name__ == '__main__':
    main()
