"""
Model 2 Full Test Evaluation Script
Computes: classification metrics, regression metrics (MAE, RMSE, R²),
          per-parameter errors, and confidence calibration (ECE).
Saves results to model_2/results/test_metrics.json
"""

import os, sys, json
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from torch.utils.data import DataLoader

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT     = r"e:\Deb\Astrolight Curve"
MODEL2   = os.path.join(ROOT, "model_2")
# model_2/data_load.py imports preprocess as a bare name — must be on sys.path
sys.path.insert(0, MODEL2)
sys.path.insert(0, ROOT)

from model_2.data_load   import ExoplanetDataset
from model_2.concantecenattion import ExoplanetLateFusionModel

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH      = os.path.join(ROOT, "modified datasets", "koi_cumulative_labeled.csv")
DATASET_DIR   = os.path.join(ROOT, "dataset")
CHECKPOINT    = os.path.join(ROOT, "model_2", "checkpoints", "best_model.pth")
RESULTS_DIR   = os.path.join(ROOT, "model_2", "results")
OUTPUT_JSON   = os.path.join(RESULTS_DIR, "test_metrics.json")
BATCH_SIZE    = 64
NUM_CLASSES   = 5
SEED          = 42

CLASS_NAMES = ["transit", "stellar_eclipse", "not_transit", "centroid_offset", "Ephemeris match"]
PARAM_NAMES = ["depth", "duration", "period", "epoch"]
# Inverse scaling functions (match train.py targets)
def inv_depth(x):    return np.expm1(x * 10.0)   # log1p(depth)/10
def inv_duration(x): return x * 10.0              # duration/10
def inv_period(x):   return np.expm1(x * 5.0)    # log1p(period)/5
def inv_epoch(x):    return np.expm1(np.abs(x) * 10.0)  # log1p(|epoch|)/10
INV_FNS = [inv_depth, inv_duration, inv_period, inv_epoch]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load & split data (same seed as training) ─────────────────────────────────
df_meta = pd.read_csv(CSV_PATH, comment='#')
label_mapping = {
    'transit': 0, 'stellar_eclipse': 1, 'not_transit': 2,
    'centroid_offset': 3, 'Ephemeris match': 4, 'unlabeled': 2
}
df_meta["label_idx"] = df_meta["signal_class"].apply(lambda x: label_mapping.get(x, 2))

train_val_df, test_df = train_test_split(
    df_meta, test_size=0.15, random_state=SEED, stratify=df_meta["label_idx"])
val_rel = 0.15 / 0.85
train_df, val_df = train_test_split(
    train_val_df, test_size=val_rel, random_state=SEED, stratify=train_val_df["label_idx"])

print(f"Test samples: {len(test_df)}")

# Compute train stats for normalisation (no leakage)
train_ds = ExoplanetDataset(train_df, DATASET_DIR, augment=False)
train_stats = {
    'medians': train_ds.stellar_medians,
    'means':   train_ds.stellar_means,
    'stds':    train_ds.stellar_stds,
}
test_ds = ExoplanetDataset(test_df, DATASET_DIR, augment=False, stats=train_stats)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ── Load model ────────────────────────────────────────────────────────────────
model = ExoplanetLateFusionModel(num_classes=NUM_CLASSES).to(device)
ckpt  = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt)
model.eval()
print("Checkpoint loaded.")

# ── Inference ─────────────────────────────────────────────────────────────────
all_labels, all_preds, all_probs = [], [], []
all_reg_pred, all_reg_true = [], []
all_conf, all_conf_label = [], []

with torch.no_grad():
    for x_global, x_local, x_stellar, y_class, y_reg, y_conf in test_loader:
        x_global  = x_global.to(device)
        x_local   = x_local.to(device)
        x_stellar = x_stellar.to(device)

        class_logits, reg_outputs, confidence, _, _ = model(x_global, x_local, x_stellar)
        probs = torch.softmax(class_logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(y_class.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_reg_pred.extend(reg_outputs.cpu().numpy())
        all_reg_true.extend(y_reg.numpy())
        all_conf.extend(confidence.cpu().squeeze().numpy().tolist()
                        if confidence.squeeze().dim() > 0
                        else [confidence.cpu().item()])
        all_conf_label.extend(y_conf.numpy().flatten().tolist())

all_labels    = np.array(all_labels)
all_preds     = np.array(all_preds)
all_probs     = np.array(all_probs)
all_reg_pred  = np.array(all_reg_pred)
all_reg_true  = np.array(all_reg_true)
all_conf      = np.array(all_conf)
all_conf_label= np.array(all_conf_label)

# ── Classification Metrics ────────────────────────────────────────────────────
acc     = accuracy_score(all_labels, all_preds)
w_f1    = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
mac_f1  = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
w_prec  = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
w_rec   = recall_score(all_labels, all_preds, average='weighted',    zero_division=0)

try:
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr',
                            labels=np.arange(NUM_CLASSES))
except Exception: roc_auc = None

per_class_prec = precision_score(all_labels, all_preds, average=None,
                                  labels=np.arange(NUM_CLASSES), zero_division=0).tolist()
per_class_rec  = recall_score(all_labels, all_preds, average=None,
                               labels=np.arange(NUM_CLASSES), zero_division=0).tolist()
per_class_f1   = f1_score(all_labels, all_preds, average=None,
                           labels=np.arange(NUM_CLASSES), zero_division=0).tolist()
per_class_supp = [int((all_labels == i).sum()) for i in range(NUM_CLASSES)]
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES))).tolist()

print(f"\nAccuracy:    {acc:.4f}")
print(f"Weighted F1: {w_f1:.4f}")
print(f"Macro F1:    {mac_f1:.4f}")
if roc_auc: print(f"ROC AUC:     {roc_auc:.4f}")

# ── Regression Metrics (transit samples only) ─────────────────────────────────
transit_mask = (all_labels == 0)
print(f"\nTransit samples in test: {transit_mask.sum()}")

reg_metrics = {}
for i, (pname, inv_fn) in enumerate(zip(PARAM_NAMES, INV_FNS)):
    pred_scaled = all_reg_pred[transit_mask, i]
    true_scaled = all_reg_true[transit_mask, i]

    # Metrics on SCALED values (direct model output, unit-free)
    mae_s  = float(np.mean(np.abs(pred_scaled - true_scaled)))
    rmse_s = float(np.sqrt(np.mean((pred_scaled - true_scaled)**2)))
    ss_res = np.sum((true_scaled - pred_scaled)**2)
    ss_tot = np.sum((true_scaled - np.mean(true_scaled))**2)
    r2_s   = float(1 - ss_res/ss_tot) if ss_tot > 0 else float('nan')

    # Metrics on PHYSICAL values (inverse-transformed)
    pred_phys = inv_fn(pred_scaled)
    true_phys = inv_fn(true_scaled)
    mae_p  = float(np.mean(np.abs(pred_phys - true_phys)))
    rmse_p = float(np.sqrt(np.mean((pred_phys - true_phys)**2)))
    ss_res_p = np.sum((true_phys - pred_phys)**2)
    ss_tot_p = np.sum((true_phys - np.mean(true_phys))**2)
    r2_p   = float(1 - ss_res_p/ss_tot_p) if ss_tot_p > 0 else float('nan')

    reg_metrics[pname] = {
        "scaled": {"mae": round(mae_s, 6), "rmse": round(rmse_s, 6), "r2": round(r2_s, 6)},
        "physical": {"mae": round(mae_p, 4), "rmse": round(rmse_p, 4), "r2": round(r2_p, 4)},
    }
    print(f"  {pname:<10}  MAE={mae_s:.4f}  RMSE={rmse_s:.4f}  R²={r2_s:.4f}  "
          f"(physical MAE={mae_p:.4f}, R²={r2_p:.4f})")

# ── Confidence Calibration (ECE) ──────────────────────────────────────────────
def expected_calibration_error(confs, labels, n_bins=10):
    """ECE: weighted average |confidence - accuracy| across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = float(labels[mask].mean())
        bin_conf = float(confs[mask].mean())
        ece += (mask.sum() / len(confs)) * abs(bin_conf - bin_acc)
    return float(ece)

ece = expected_calibration_error(all_conf, all_conf_label)
mean_conf     = float(all_conf.mean())
mean_conf_acc = float(all_conf_label.mean())   # fraction of transits in test
print(f"\nECE:            {ece:.4f}")
print(f"Mean confidence:{mean_conf:.4f}  (true transit fraction: {mean_conf_acc:.4f})")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "classification": {
        "accuracy":          round(acc,    4),
        "weighted_f1":       round(w_f1,   4),
        "macro_f1":          round(mac_f1, 4),
        "weighted_precision":round(w_prec, 4),
        "weighted_recall":   round(w_rec,  4),
        "roc_auc_ovr":       round(roc_auc, 4) if roc_auc else None,
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(per_class_prec[i], 4),
                "recall":    round(per_class_rec[i],  4),
                "f1":        round(per_class_f1[i],   4),
                "support":   per_class_supp[i],
            } for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm,
    },
    "regression": reg_metrics,
    "confidence_calibration": {
        "ece":                 round(ece,          4),
        "mean_confidence":     round(mean_conf,    4),
        "true_transit_frac":   round(mean_conf_acc,4),
    }
}

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n[DONE] Saved to {OUTPUT_JSON}")
