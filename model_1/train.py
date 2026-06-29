import os
import json
import torch
import numpy as np
import pandas as pd

from model_1.config import Config
from model_1.dataset import get_data_loaders
from model_1.model import ExoplanetDeepModel
from model_1.trainer import ExoplanetTrainer
from model_1.metrics import ModelEvaluator
from model_1.utils import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_attention_heatmap,
    plot_example_predicted_transit,
    plot_regression_predictions
)

def run_pipeline(config=Config):
    # 1. Load DataLoaders
    print("Loading data splits...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders(config=config)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Calculate class weights for focal / cross entropy loss
    # Access training DataFrame directly via dataset reference
    train_df = train_loader.dataset.df_index
    class_counts = train_df["label_idx"].value_counts()
    
    total_samples = len(train_df)
    class_weights = torch.zeros(config.NUM_CLASSES)
    for c in range(config.NUM_CLASSES):
        count = class_counts.get(c, 0)
        class_weights[c] = total_samples / (config.NUM_CLASSES * count) if count > 0 else 1.0
        
    print(f"Class counts in training set: {class_counts.to_dict()}")
    print(f"Computed class weights: {class_weights.tolist()}")

    # 2. Build Model
    print("Building Exoplanet Deep Model...")
    model = ExoplanetDeepModel(config=config)
    
    # 3. Initialize Trainer
    trainer = ExoplanetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config,
        device='cuda'
    )

    # 4. Train Model
    history = trainer.fit()
    
    # Plot learning curves
    plot_learning_curves(history, output_dir=config.RESULTS_DIR)
    print(f"Saved training learning curves to {config.RESULTS_DIR}")

    # 5. Evaluate on Test Split
    print("Evaluating on Test Split...")
    model.eval()
    
    y_true_class = []
    y_pred_probs = []
    y_true_reg = []
    y_pred_reg = []
    y_true_conf = []
    y_pred_conf = []
    
    # Save one positive transit sample for attention visualization
    transit_sample = None
    
    with torch.no_grad():
        for x, y_class, y_reg, y_conf in test_loader:
            x = x.to(trainer.device)
            
            # Forward pass
            class_logits, reg_outputs, confidence, attn_weights = model(x)
            
            probs = torch.softmax(class_logits, dim=1).cpu().numpy()
            reg = reg_outputs.cpu().numpy()
            conf = confidence.cpu().numpy()
            
            y_true_class.extend(y_class.numpy())
            y_pred_probs.extend(probs)
            y_true_reg.extend(y_reg.numpy())
            y_pred_reg.extend(reg)
            y_true_conf.extend(y_conf.numpy())
            y_pred_conf.extend(conf)
            
            # Extract first positive transit sample in the batch for visualization
            if transit_sample is None:
                transits = (y_class == 0).nonzero(as_tuple=True)[0]
                if len(transits) > 0:
                    idx = transits[0].item()
                    transit_sample = {
                        'flux': x[idx, 0].cpu().numpy(),
                        'true_class': int(y_class[idx].item()),
                        'pred_class': int(np.argmax(probs[idx])),
                        'pred_conf': float(conf[idx, 0]),
                        'attn_weights': attn_weights[idx].cpu().numpy()
                    }

    y_true_class = np.array(y_true_class)
    y_pred_probs = np.array(y_pred_probs)
    y_true_reg = np.array(y_true_reg)
    y_pred_reg = np.array(y_pred_reg)
    y_true_conf = np.array(y_true_conf).flatten()
    y_pred_conf = np.array(y_pred_conf).flatten()

    # 6. Compute Metrics
    class_metrics = ModelEvaluator.evaluate_classification(y_true_class, y_pred_probs, config.NUM_CLASSES)
    reg_metrics = ModelEvaluator.evaluate_regression(y_true_reg, y_pred_reg)
    ece = ModelEvaluator.compute_ece(y_pred_conf, y_true_conf, n_bins=10)

    # Compile all metrics
    metrics_summary = {
        'classification': class_metrics,
        'regression': reg_metrics,
        'expected_calibration_error': ece
    }

    # Print Classification Results
    print("\n" + "="*50)
    print("FINAL TEST METRICS")
    print("="*50)
    print(f"Accuracy:                  {class_metrics['accuracy']:.4f}")
    print(f"Weighted F1-score:         {class_metrics['f1_score']:.4f}")
    print(f"Weighted Precision:        {class_metrics['precision']:.4f}")
    print(f"Weighted Recall:           {class_metrics['recall']:.4f}")
    print(f"Multiclass ROC AUC:        {class_metrics['roc_auc']:.4f}")
    print(f"Multiclass PR AUC:         {class_metrics['pr_auc']:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    
    print("\nRegression Metrics (Scaled Outputs):")
    for name, m in reg_metrics.items():
        if name != 'average':
            print(f"  - {name.title()}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}")
    print("="*50)

    # Save metrics JSON file
    metrics_file = os.path.join(config.RESULTS_DIR, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved metrics summary JSON file to {metrics_file}")

    # 7. Generate Visualizations
    print("Generating validation plot suite...")
    plot_confusion_matrix(class_metrics['confusion_matrix'], config.CLASS_NAMES, config.RESULTS_DIR)
    plot_roc_pr_curves(y_true_class, y_pred_probs, config.CLASS_NAMES, config.RESULTS_DIR)
    plot_regression_predictions(y_true_reg, y_pred_reg, config.RESULTS_DIR)

    # Plot attention map and example predicted transit if sample found
    if transit_sample is not None:
        plot_attention_heatmap(
            transit_sample['flux'], 
            transit_sample['attn_weights'], 
            config.RESULTS_DIR, 
            filename="example_transit_attention_map.png"
        )
        plot_example_predicted_transit(
            transit_sample['flux'],
            transit_sample['true_class'],
            transit_sample['pred_class'],
            transit_sample['pred_conf'],
            config.RESULTS_DIR
        )
        print("Successfully generated example transit plots with attention map overlays.")
    else:
        print("Warning: No positive transit samples found in test split for overlay plots.")

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
