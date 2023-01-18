import io
import os
import random
import shutil
import tarfile
import zipfile
from datetime import date
from typing import Tuple, List

import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATA_DIR = 'data/caltech101/'
CALTECH_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"


def download_data_if_not_exists():
    if not os.path.exists(DATA_DIR):
        r = requests.get(CALTECH_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        with tarfile.open("caltech-101/101_ObjectCategories.tar.gz") as file:
            file.extractall()
        os.rename("101_ObjectCategories", "caltech101")
        shutil.move("caltech101", DATA_DIR)
        shutil.rmtree("__MACOSX")
        shutil.rmtree("caltech-101")


def choose_biggest_classes(data_path: str, n_classes: int) -> Tuple[List[str], int]:
    classes = []
    for c in os.listdir(data_path):
        if c not in ('BACKGROUND_Google', 'Faces_easy'):
            classes.append((len(os.listdir(os.path.join(data_path, c))), c))

    classes.sort(reverse=True)
    classes = classes[:n_classes]
    samples_min = classes[-1][0]
    classes = [c[1] for c in classes]
    return classes, samples_min


def load_data(data_dir, classes, samples_per_cat=None):
    images, labels = [], []
    for c in classes:
        dir_path = os.path.join(data_dir, c)
        images_in_dir = os.listdir(dir_path)
        if samples_per_cat:
            images_in_dir = random.sample(images_in_dir, samples_per_cat)
        for file in images_in_dir:
            image = Image.open(os.path.join(dir_path, file)).convert('RGB')
            images.append(np.asarray(image))
            labels.append(c)

    return images, labels


def resize_images(images):
    resized = np.array([np.asarray(Image.fromarray(i).resize((300, 200))) for i in images])
    return resized


def standardize_images(images):
    images = np.array(images, dtype=np.float32)
    mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
    std = np.std(images, axis=(0, 1, 2), keepdims=True)
    return (images - mean) / std, mean, std


def display_standardized_image(image: np.ndarray, mean, std):
    image = image * std[0] + mean[0]
    image = Image.fromarray(image.astype(np.uint8))
    display(image)


def one_hot_encoding_for_labels(labels):
    labels_enc = LabelEncoder()
    labels_enc.fit(labels)
    y = labels_enc.transform(labels)

    y = y.reshape((-1, 1))

    labels_ohe = OneHotEncoder()
    labels_ohe.fit(y)
    y = labels_ohe.transform(y)

    return y.toarray(), labels_enc, labels_ohe


def resolve_label(label, labels_enc, labels_ohe):
    label = labels_ohe.inverse_transform(label.reshape(1, -1))
    label = labels_enc.inverse_transform(label[0])
    return label[0]


def count_samples_in_class(dataset, labels_enc, labels_ohe):
    d = {}
    for i in dataset:
        i = resolve_label(i, labels_enc, labels_ohe)
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    print(d)


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(200, 300, 3)),

        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(15, activation="softmax")
    ])
    model.compile(optimizer='Adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, restore_best_weights=True)

    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_data=(X_val, y_val), callbacks=[early_stopping])


def plot_metrics_from_model_history(history, metric: str):
    plt.plot(history.epoch, history.history[metric], 'g', label=f'{metric} on training set')
    plt.plot(history.epoch, history.history['val_' + metric], 'r', label=f'{metric} on test set')
    plt.title(f'Training {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()


def plot_confusion_matrix(confusion_matrix, labels):
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot()
    cax = ax.matshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(cax)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.show()


def show_classification_example(raw_predictions, img, prediction_label, true_label, labels_enc, display_image):
    pp = {labels_enc.inverse_transform([i])[0]: pi for i, pi in enumerate(raw_predictions)}
    if prediction_label == true_label:
        print(f"{prediction_label} correctly classified")
    else:
        print(f"{true_label} classified as {prediction_label}")
    display_image(img)
    print("probabilities:")
    for i, pi in sorted(pp.items(), key=lambda x: x[1], reverse=True):
        print(f"{i}: {pi}")


def save_model(model, prefix):
    model_path = f'models/{prefix}-{date.today()}'
    model.save(model_path)
    return model_path


def convert_predictions(y_predictions, y_true, labels_ohe):
    return np.asarray(tf.argmax(y_predictions.T)), labels_ohe.inverse_transform(y_true).reshape((1, -1))[0]


def get_labels_from_predictions(y_test_predictions, y_test_true, labels_enc):
    return labels_enc.inverse_transform([i for i in y_test_predictions]), labels_enc.inverse_transform([i for i in y_test_true]),
