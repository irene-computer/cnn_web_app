import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import io
import base64
from config import Config

class Visualizer:
    """Classe pour la visualisation des résultats"""
    
    @staticmethod
    def plot_training_history(history):
        """Affiche les courbes d'apprentissage"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        best_val_loss = min(history.history['val_loss'])
        best_val_acc = max(history.history['val_accuracy'])
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        return best_val_loss, best_val_acc
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """Affiche la matrice de confusion"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix - CNN on CIFAR-10', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("\nClassification Report:")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=Config.CLASS_NAMES))
        return cm
    
    @staticmethod
    def plot_misclassified_images(x_test, y_true, y_pred, num_images=10):
        """Affiche les images mal classifiées"""
        misclassified_idx = np.where(y_true != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            print("Toutes les images sont bien classifiées!")
            return
        
        num_images = min(num_images, len(misclassified_idx))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(num_images):
            idx = misclassified_idx[i]
            axes[i].imshow(x_test[idx])
            axes[i].set_title(f"True: {Config.CLASS_NAMES[y_true[idx]]}\nPred: {Config.CLASS_NAMES[y_pred[idx]]}")
            axes[i].axis('off')
        
        plt.suptitle(f'Misclassified Examples ({num_images} images)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_training_plots_base64(history):
        """Crée les courbes d'apprentissage en base64 pour l'affichage web"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(history['loss'], label='Training Loss', linewidth=2, color='#3b82f6')
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#ef4444')
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2, color='#3b82f6')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#ef4444')
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return plot