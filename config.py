import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-12345'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    MODEL_PATH = 'models/cnn_cifar10_final.keras'
    HISTORY_PATH = 'models/training_history.pkl'
    
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    EPOCHS = 30
    
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    CLASS_INFO = {
        'airplane': {'icon': '✈️', 'description': 'Aircraft, jets, planes'},
        'automobile': {'icon': '🚗', 'description': 'Cars, sedans, SUVs'},
        'bird': {'icon': '🐦', 'description': 'Various bird species'},
        'cat': {'icon': '🐱', 'description': 'Domestic cats'},
        'deer': {'icon': '🦌', 'description': 'Deer, antlered animals'},
        'dog': {'icon': '🐶', 'description': 'Domestic dogs'},
        'frog': {'icon': '🐸', 'description': 'Amphibians'},
        'horse': {'icon': '🐴', 'description': 'Equine animals'},
        'ship': {'icon': '🚢', 'description': 'Boats, vessels, ships'},
        'truck': {'icon': '🚚', 'description': 'Trucks, heavy vehicles'}
    }