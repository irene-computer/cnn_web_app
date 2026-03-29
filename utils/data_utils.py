import tensorflow as tf
from tensorflow import keras
import numpy as np
from config import Config

def load_and_preprocess_data():
    """Charge et prétraite les données CIFAR-10"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test

def create_tf_dataset(x, y, batch_size=Config.BATCH_SIZE, shuffle=True):
    """Crée un dataset TensorFlow pour l'entraînement"""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_cifar10_info():
    """Retourne les informations sur le dataset"""
    return {
        'train_size': 50000,
        'test_size': 10000,
        'image_shape': (32, 32, 3),
        'num_classes': 10,
        'class_names': Config.CLASS_NAMES
    }