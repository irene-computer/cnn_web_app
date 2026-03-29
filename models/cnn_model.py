import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import Config

class CustomCNN(keras.Model):
    """
    Modèle CNN personnalisé pour la classification d'images CIFAR-10
    Architecture:
    - 3 blocs de convolution avec MaxPooling
    - Data Augmentation intégrée
    - Classificateur MLP avec Dropout
    """
    
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(CustomCNN, self).__init__()
        
        # Data Augmentation (appliquée pendant l'entraînement)
        self.data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Premier bloc de convolution
        self.conv1_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.conv1_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        # Deuxième bloc de convolution
        self.conv2_1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv2_2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        # Troisième bloc de convolution
        self.conv3_1 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv3_2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        # Classificateur MLP
        self.flatten = layers.Flatten()
        self.dropout1 = layers.Dropout(0.5)
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Data Augmentation (uniquement pendant l'entraînement)
        if training:
            x = self.data_augmentation(x)
        
        # Bloc 1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        # Bloc 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        # Bloc 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        
        # Classificateur
        x = self.flatten(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        
        return x
    
    def build_graph(self):
        """Construit le graphe du modèle pour l'affichage"""
        x = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))


def create_model():
    """Crée et compile le modèle CNN"""
    model = CustomCNN()
    
    # Construire le modèle avec une entrée fictive
    model.build((None, Config.IMG_HEIGHT, Config.IMG_WIDTH, 3))
    
    return model


def get_model_summary():
    """Retourne le résumé du modèle"""
    model = create_model()
    return model.summary()