import numpy as np
import pickle
from tensorflow import keras
from config import Config
from utils.data_utils import load_and_preprocess_data
from utils.visualization import Visualizer

print("=" * 60)
print("EVALUATING CNN MODEL")
print("=" * 60)

print("\n1. Loading model...")
try:
    model = keras.models.load_model(Config.MODEL_PATH)
    print(f"✅ Model loaded: {Config.MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

print("\n2. Loading test data...")
_, _, x_test, y_test = load_and_preprocess_data()
y_true = y_test.flatten()

print("\n3. Evaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\n📊 Test Loss: {test_loss:.4f}")
print(f"📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

print("\n4. Generating predictions...")
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n5. Confusion Matrix...")
Visualizer.plot_confusion_matrix(y_true, y_pred_classes)

print("\n6. Misclassified examples...")
Visualizer.plot_misclassified_images(x_test, y_true, y_pred_classes, num_images=10)

print("\n7. Loading training history...")
try:
    with open(Config.HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)
    print("✅ Training history loaded")
    Visualizer.plot_training_history_from_dict = lambda h: Visualizer.plot_training_history(type('obj', (object,), {'history': h})())
except Exception as e:
    print(f"⚠️ Training history not found: {e}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)