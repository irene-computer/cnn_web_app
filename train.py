import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("=" * 60)
print("ENTRAÎNEMENT DU MODÈLE CNN SUR CIFAR-10")
print("=" * 60)

# 1. Chargement des données
print("\n1. Chargement des données CIFAR-10...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalisation
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"Train: {x_train.shape}, Test: {x_test.shape}")
print(f"Classes: {len(np.unique(y_train))}")

# 2. Création du modèle CNN avec API Fonctionnelle
print("\n2. Création du modèle CNN...")

inputs = keras.Input(shape=(32, 32, 3))

# Data Augmentation
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)

# Bloc 1
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Bloc 2
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Bloc 3
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Classificateur
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Afficher le résumé du modèle
print("\n" + "=" * 60)
model.summary()
print("=" * 60)

# 3. Callbacks
print("\n3. Configuration des callbacks...")

# EarlyStopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ModelCheckpoint
os.makedirs('models', exist_ok=True)
checkpoint = keras.callbacks.ModelCheckpoint(
    'models/cnn_cifar10_final.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ReduceLROnPlateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]

# 4. Entraînement
print("\n4. Début de l'entraînement...")
print("=" * 60)

history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=30,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Entraînement terminé après {len(history.history['loss'])} epochs")

# 5. Évaluation
print("\n5. Évaluation sur le jeu de test...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

# Vérification de l'objectif
if test_acc >= 0.70:
    print("\n✅ OBJECTIF ATTEINT! Précision >= 70%")
else:
    print(f"\n⚠️ OBJECTIF NON ATTEINT. Précision: {test_acc:.4f} < 0.70")

# 6. Sauvegarde
print("\n6. Sauvegarde du modèle...")
model.save('models/cnn_cifar10_final.keras')
print(f"✅ Modèle sauvegardé: models/cnn_cifar10_final.keras")

with open('models/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print(f"✅ Historique sauvegardé: models/training_history.pkl")

print("\n" + "=" * 60)
print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("=" * 60)
