import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

class Visualizer:
    @staticmethod
    def plot_confusion_matrix(y_test, pred, label_encoder, logger, log_file_name, fold_num):
        """
        confusion matrix 시각화
        """
        report_str = classification_report(y_test, pred, zero_division=1)
        logger.info(f"Classification Report:\n{report_str}")
        
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        image_path = log_file_name.replace('.log', f'_confusion_matrix{fold_num}.png')
        plt.savefig(image_path)
        logger.info(f"Confusion matrix saved to {image_path}")
        plt.close()
    
    @staticmethod
    def plot_learning_curve(model, df_x, df_y, logger, log_file_name):
        """
        learning curve 시각화
        """
        logger.info("Generating learning curve...")

        cv = StratifiedKFold(n_splits=5)

        train_sizes, train_scores, val_scores = learning_curve(
            model,
            df_x, df_y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_mean, label="Training Accuracy", marker='o')
        plt.plot(train_sizes, val_mean, label="Validation Accuracy", marker='s')
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid()
        
        image_path = log_file_name.replace('.log', '_learning_curve.png')
        plt.savefig(image_path)
        logger.info(f"Learning curve saved to {image_path}")
        plt.close()
