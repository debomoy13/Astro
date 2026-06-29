import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from model_1.config import Config

# Setup matplotlib style for clean research publication visuals
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

def plot_learning_curves(history, output_dir=Config.RESULTS_DIR):
    """
    Plots training and validation loss curves and classification accuracy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss Curve
    axes[0].plot(history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    axes[0].set_title('Multi-Task Learning Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend(frameon=True)
    
    # 2. Accuracy Curve
    axes[1].plot(history['train_class_acc'], label='Train Class Acc', color='#2ca02c', linewidth=2)
    axes[1].plot(history['val_class_acc'], label='Val Class Acc', color='#d62728', linewidth=2, linestyle='--')
    axes[1].set_title('Classification Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()


def plot_confusion_matrix(cm, class_names=Config.CLASS_NAMES, output_dir=Config.RESULTS_DIR):
    """
    Plots the classification confusion matrix.
    """
    cm = np.array(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them with the respective list entries
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, 
        yticklabels=class_names,
        title='Normalized Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm_normalized[i, j], fmt) + f"\n({cm[i, j]})",
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black"
            )
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def plot_roc_pr_curves(y_true, y_pred_probs, class_names=Config.CLASS_NAMES, output_dir=Config.RESULTS_DIR):
    """
    Plots One-vs-Rest ROC curves and Precision-Recall curves.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.array(y_pred_probs, dtype=float)
    num_classes = len(class_names)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. ROC Curves
    for c in range(num_classes):
        y_true_binary = (y_true == c).astype(int)
        if np.sum(y_true_binary) > 0 and np.sum(y_true_binary) < len(y_true):
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, c])
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, label=f"{class_names[c]} (AUC = {roc_auc:.3f})", color=colors[c], linewidth=2)
            
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('One-vs-Rest ROC Curves')
    axes[0].legend(loc="lower right", frameon=True)
    
    # 2. PR Curves
    for c in range(num_classes):
        y_true_binary = (y_true == c).astype(int)
        if np.sum(y_true_binary) > 0 and np.sum(y_true_binary) < len(y_true):
            p, r, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, c])
            pr_auc = auc(r, p)
            axes[1].plot(r, p, label=f"{class_names[c]} (AUC = {pr_auc:.3f})", color=colors[c], linewidth=2)
            
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('One-vs-Rest Precision-Recall Curves')
    axes[1].legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_pr_curves.png'))
    plt.close()


def plot_attention_heatmap(flux, attention_weights, output_dir=Config.RESULTS_DIR, filename="attention_heatmap.png"):
    """
    Visualizes attention weights mapping back to the light curve structure.
    """
    flux = np.array(flux).flatten()
    attn = np.array(attention_weights)
    
    # If multihead attention was used, attn is (T_pooled, T_pooled)
    # Average across target sequence dimension to get a 1D importance score per pooled time step
    if len(attn.shape) == 2:
        attn = np.mean(attn, axis=0)
    
    # Interpolate pooled attention weights back to original sequence length
    attn_interp = np.interp(
        np.linspace(0, len(attn)-1, len(flux)),
        np.arange(len(attn)),
        attn
    )
    
    # Normalize attention weights for visual shading (0 to 1)
    if np.max(attn_interp) - np.min(attn_interp) > 0:
        attn_norm = (attn_interp - np.min(attn_interp)) / (np.max(attn_interp) - np.min(attn_interp))
    else:
        attn_norm = np.zeros_like(attn_interp)

    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot the normalized flux
    time_steps = np.arange(len(flux))
    ax.plot(time_steps, flux, color='black', alpha=0.8, linewidth=1.5, label='Light Curve')
    
    # Overlay attention shading
    for i in range(len(flux) - 1):
        # Scale alpha based on attention importance
        ax.axvspan(i, i+1, color='orange', alpha=float(attn_norm[i] * 0.45), ymin=0, ymax=1)
        
    ax.set_title('Temporal Self-Attention Map Overlay')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Normalized Flux')
    
    # Dummy plot to create legend entry for attention
    ax.fill_between([0], [0], [0], color='orange', alpha=0.4, label='Attention Weight')
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_example_predicted_transit(flux, true_class, pred_class, pred_conf, output_dir=Config.RESULTS_DIR):
    """
    Plots an example light curve alongside true vs predicted class labels.
    """
    flux = np.array(flux).flatten()
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(flux, color='#1f77b4', linewidth=1.5)
    ax.set_title('Example Light Curve Inference result')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Relative Flux')
    
    # Label formatting
    info_text = (
        f"Ground Truth: {Config.CLASS_NAMES[true_class]}\n"
        f"Predicted Class: {Config.CLASS_NAMES[pred_class]}\n"
        f"Transit Confidence: {pred_conf:.2%}"
    )
    
    # Place text box in the upper right
    ax.text(
        0.05, 0.05, info_text, 
        transform=ax.transAxes, 
        verticalalignment='bottom', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_prediction.png'))
    plt.close()


def plot_regression_predictions(y_true, y_pred, output_dir=Config.RESULTS_DIR):
    """
    Scatter plots comparing actual vs predicted transit parameters:
    Transit depth and Orbital period.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    if len(y_true) == 0:
        return
        
    # Scale back log-scaled parameters for realistic plotting
    # y_reg format: [depth_scaled, duration_scaled, period_scaled, midpoint_scaled]
    # depth = expm1(depth_scaled * 10)
    # period = expm1(period_scaled * 5)
    true_depths = np.expm1(y_true[:, 0] * 10.0)
    pred_depths = np.expm1(y_pred[:, 0] * 10.0)
    
    true_periods = np.expm1(y_true[:, 2] * 5.0)
    pred_periods = np.expm1(y_pred[:, 2] * 5.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Depth comparison
    axes[0].scatter(true_depths, pred_depths, alpha=0.7, color='#1f77b4', edgecolors='k')
    max_val = max(np.max(true_depths), np.max(pred_depths))
    min_val = min(np.min(true_depths), np.min(pred_depths))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
    axes[0].set_xlabel('Actual Depth (ppm)')
    axes[0].set_ylabel('Predicted Depth (ppm)')
    axes[0].set_title('Transit Depth Estimation Accuracy')
    
    # 2. Period comparison
    axes[1].scatter(true_periods, pred_periods, alpha=0.7, color='#2ca02c', edgecolors='k')
    max_val = max(np.max(true_periods), np.max(pred_periods))
    min_val = min(np.min(true_periods), np.min(pred_periods))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
    axes[1].set_xlabel('Actual Period (days)')
    axes[1].set_ylabel('Predicted Period (days)')
    axes[1].set_title('Orbital Period Estimation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_predictions.png'))
    plt.close()
