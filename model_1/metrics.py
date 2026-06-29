import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    confusion_matrix, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

class ModelEvaluator:
    """
    Evaluates multi-task model performance for classification, regression, 
    and confidence scores.
    """

    @staticmethod
    def compute_ece(probs, labels, n_bins=10):
        """
        Computes the Expected Calibration Error (ECE) for confidence calibration.
        ECE measures the discrepancy between predicted confidence scores and true accuracy.
        """
        probs = np.array(probs, dtype=float).flatten()
        labels = np.array(labels, dtype=float).flatten()
        
        # Define bin boundaries
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_samples = len(probs)
        
        if n_samples == 0:
            return 0.0
            
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i+1]
            
            # Find indices falling within the bin [bin_lower, bin_upper)
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                # Compute accuracy (fraction of matches) in this bin
                bin_accuracy = np.mean(labels[in_bin])
                # Compute average confidence in this bin
                bin_confidence = np.mean(probs[in_bin])
                # Weight by bin size relative to total size
                ece += (bin_size / n_samples) * np.abs(bin_confidence - bin_accuracy)
                
        return float(ece)

    @classmethod
    def evaluate_classification(cls, y_true, y_pred_probs, num_classes=5):
        """
        Computes classification performance metrics.
        Args:
            y_true: (N,) true integer class labels
            y_pred_probs: (N, num_classes) predicted class probabilities
        """
        y_true = np.array(y_true, dtype=int)
        y_pred_probs = np.array(y_pred_probs, dtype=float)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        # Basic scores
        acc = accuracy_score(y_true, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average='weighted', zero_division=0
        )
        
        # Per-class scores
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average=None, labels=np.arange(num_classes), zero_division=0
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes, labels=np.arange(num_classes))

        # Multi-class ROC AUC (One-Vs-Rest)
        try:
            roc_auc = roc_auc_score(
                y_true, 
                y_pred_probs, 
                multi_class='ovr', 
                labels=np.arange(num_classes)
            )
        except Exception:
            # Fallback if a class is not represented in the batch
            roc_auc = 0.5

        # Multi-class PR AUC (One-Vs-Rest)
        pr_aucs = []
        for c in range(num_classes):
            y_true_binary = (y_true == c).astype(int)
            # Check if class exists in ground truth to avoid NaN
            if np.sum(y_true_binary) > 0 and np.sum(y_true_binary) < len(y_true):
                y_pred_binary = y_pred_probs[:, c]
                p, r, _ = precision_recall_curve(y_true_binary, y_pred_binary)
                pr_aucs.append(auc(r, p))
            else:
                pr_aucs.append(0.0)
                
        pr_auc_macro = float(np.mean(pr_aucs))

        return {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': pr_auc_macro,
            'class_precision': class_precision.tolist(),
            'class_recall': class_recall.tolist(),
            'class_f1': class_f1.tolist(),
            'confusion_matrix': cm.tolist()
        }

    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """
        Computes regression metrics for transit parameters.
        Args:
            y_true: (N, 4) -> [depth, duration, period, midpoint] (in scaled/raw coordinates)
            y_pred: (N, 4)
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        
        if len(y_true) == 0:
            return {}

        param_names = ['depth', 'duration', 'period', 'midpoint']
        metrics = {}

        for i, name in enumerate(param_names):
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            
            # R-squared (coefficient of determination)
            try:
                r2 = r2_score(y_true[:, i], y_pred[:, i])
            except Exception:
                r2 = 0.0
                
            metrics[name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            }

        # Average regression scores
        metrics['average'] = {
            'mae': float(np.mean([metrics[n]['mae'] for n in param_names])),
            'rmse': float(np.mean([metrics[n]['rmse'] for n in param_names])),
            'r2': float(np.mean([metrics[n]['r2'] for n in param_names]))
        }

        return metrics
