import os
import pickle
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow import keras

from config import Config
from utils.visualization import Visualizer

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variables globales
model = None
training_history = None
x_test = None
y_test = None

def load_model():
    """Charge le modèle sauvegardé"""
    global model
    try:
        if os.path.exists(Config.MODEL_PATH):
            model = keras.models.load_model(Config.MODEL_PATH)
            print(f"✅ Model loaded: {Config.MODEL_PATH}")
            return True
        else:
            print(f"⚠️ Model not found: {Config.MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def load_cifar10():
    """Charge les données CIFAR-10 pour les tests"""
    global x_test, y_test
    try:
        (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        print(f"✅ CIFAR-10 loaded: {len(x_test)} test images")
        return True
    except Exception as e:
        print(f"❌ Error loading CIFAR-10: {e}")
        return False

def load_history():
    """Charge l'historique d'entraînement"""
    global training_history
    try:
        if os.path.exists(Config.HISTORY_PATH):
            with open(Config.HISTORY_PATH, 'rb') as f:
                training_history = pickle.load(f)
            print("✅ Training history loaded")
    except Exception as e:
        print(f"⚠️ Could not load training history: {e}")

def allowed_file(filename):
    """Vérifie si le fichier est autorisé"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Prétraite l'image pour le modèle"""
    image = image.resize((32, 32))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    """Fait une prédiction sur une image"""
    if model is None:
        raise Exception("Model not loaded")
    
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]
    
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])
    
    top_5_indices = np.argsort(predictions)[-5:][::-1]
    top_5_predictions = [
        {
            'class': Config.CLASS_NAMES[i],
            'confidence': float(predictions[i]),
            'percentage': f"{predictions[i] * 100:.1f}%"
        }
        for i in top_5_indices
    ]
    
    return {
        'predicted_class': Config.CLASS_NAMES[predicted_idx],
        'confidence': confidence,
        'confidence_percentage': f"{confidence * 100:.1f}%",
        'top_5': top_5_predictions
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html', class_names=Config.CLASS_NAMES, class_info=Config.CLASS_INFO)

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/test')
def test_page():
    return render_template('dashboard.html')

@app.route('/performance')
def performance_page():
    loss_plot = None
    metrics = None
    if training_history:
        loss_plot = Visualizer.create_training_plots_base64(training_history)
        metrics = {
            'best_val_acc': max(training_history['val_accuracy']) * 100,
            'best_val_loss': min(training_history['val_loss']),
            'final_val_acc': training_history['val_accuracy'][-1] * 100,
            'final_val_loss': training_history['val_loss'][-1],
            'epochs': len(training_history['loss'])
        }
    return render_template('performance.html', loss_plot=loss_plot, metrics=metrics)

@app.route('/predict/upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return render_template('predict.html', error='No file provided')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('predict.html', error='Empty filename')
    
    if not allowed_file(file.filename):
        return render_template('predict.html', error='File type not allowed. Use PNG, JPG, JPEG, GIF, BMP')
    
    try:
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = Image.open(filepath)
        result = predict_image(image)
        result['image_path'] = url_for('static', filename=f'uploads/{filename}')
        
        return render_template('predict.html', result=result)
    except Exception as e:
        return render_template('predict.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        image = Image.open(file)
        result = predict_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test/random')
def test_random():
    try:
        if x_test is None or y_test is None:
            if not load_cifar10():
                return jsonify({'error': 'Could not load CIFAR-10 dataset'}), 500
        
        idx = np.random.randint(0, len(x_test))
        img = x_test[idx]
        label = Config.CLASS_NAMES[int(y_test[idx][0])]
        image = Image.fromarray((img * 255).astype('uint8'))
        result = predict_image(image)
        result['true_class'] = label
        result['is_correct'] = result['predicted_class'] == label
        result['image_data'] = (img * 255).astype('uint8').tolist()
        result['index'] = idx
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test/specific/<int:index>')
def test_specific(index):
    try:
        if x_test is None or y_test is None:
            if not load_cifar10():
                return jsonify({'error': 'Could not load CIFAR-10 dataset'}), 500
        
        test_size = len(x_test)
        
        if index < 0 or index >= test_size:
            return jsonify({'error': f'Index {index} invalid. Use a number between 0 and {test_size-1}'}), 400
        
        img = x_test[index]
        label = Config.CLASS_NAMES[int(y_test[index][0])]
        image = Image.fromarray((img * 255).astype('uint8'))
        result = predict_image(image)
        result['true_class'] = label
        result['is_correct'] = result['predicted_class'] == label
        result['image_data'] = (img * 255).astype('uint8').tolist()
        result['index'] = index
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test/batch', methods=['POST'])
def test_batch():
    try:
        if x_test is None or y_test is None:
            if not load_cifar10():
                return jsonify({'error': 'Could not load CIFAR-10 dataset'}), 500
        
        data = request.get_json()
        indices = data.get('indices', [])
        results = []
        
        for idx in indices:
            if 0 <= idx < len(x_test):
                img = x_test[idx]
                label = Config.CLASS_NAMES[int(y_test[idx][0])]
                image = Image.fromarray((img * 255).astype('uint8'))
                pred = predict_image(image)
                results.append({
                    'index': idx,
                    'true_class': label,
                    'predicted_class': pred['predicted_class'],
                    'confidence': pred['confidence'],
                    'is_correct': pred['predicted_class'] == label
                })
        
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = (correct / len(results)) * 100 if results else 0
        
        return jsonify({
            'results': results,
            'total': len(results),
            'correct': correct,
            'accuracy': accuracy
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CNN VISION - IMAGE CLASSIFICATION APPLICATION")
    print("=" * 60)
    
    load_model()
    load_cifar10()
    load_history()
    
    print(f"\n📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🤖 Model: {Config.MODEL_PATH}")
    if x_test is not None:
        print(f"📊 Test images available: 0 to {len(x_test)-1}")
    print(f"🌐 Access at: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
