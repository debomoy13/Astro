import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model_1.config import Config
from model_1.preprocessing import LightCurvePreprocessor
from model_1.candidate_detection import TransitCandidateDetector
from model_1.augmentations import TimeSeriesAugmenter

class ExoplanetDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Kepler light curves, matching with
    KOI catalog entries to fetch labels & regression parameters, and applying
    preprocessing and data augmentations.
    """
    def __init__(self, df_index, df_meta, config=Config, augment=False, use_bls_windows=None):
        self.df_index = df_index.reset_index(drop=True)
        self.df_meta = df_meta
        self.config = config
        self.augment = augment
        
        # Override config default if specified
        self.use_bls_windows = use_bls_windows if use_bls_windows is not None else config.USE_BLS_WINDOWS
        self.target_len = config.BLS_WINDOW_SIZE if self.use_bls_windows else config.SEQUENCE_LENGTH

    def __len__(self):
        return len(self.df_index)

    def __getitem__(self, idx):
        row = self.df_index.iloc[idx]
        kepid = int(row["kepid"])
        label_str = row["label"]
        file_path = row["file"]
        
        # Fix Windows/Unix path separators
        file_path = file_path.replace("\\", "/")
        full_path = os.path.join(self.config.BASE_DIR, file_path)
        if not os.path.exists(full_path):
            # Try absolute path directly
            full_path = file_path
            
        # 1. Load light curve data
        try:
            data = np.load(full_path)
            time = data["time"]
            flux = data["flux"]
        except Exception as e:
            # Fallback in case of corruption/missing file
            time = np.linspace(0, 10, self.config.SEQUENCE_LENGTH)
            flux = np.ones(self.config.SEQUENCE_LENGTH)

        # 2. Extract classification, regression and confidence labels
        # Find matching metadata row in cumulative catalog
        meta_matches = self.df_meta[self.df_meta["kepid"] == kepid]
        if len(meta_matches) > 0:
            meta_row = meta_matches.iloc[0]
            # Replace NaNs in target parameters with 0.0
            depth = float(meta_row["koi_depth"]) if not pd.isna(meta_row["koi_depth"]) else 0.0
            duration = float(meta_row["koi_duration"]) if not pd.isna(meta_row["koi_duration"]) else 0.0
            period = float(meta_row["koi_period"]) if not pd.isna(meta_row["koi_period"]) else 0.0
            time0bk = float(meta_row["koi_time0bk"]) if not pd.isna(meta_row["koi_time0bk"]) else 0.0
        else:
            # Fallback if no matching KIC in cumulative metadata
            depth, duration, period, time0bk = 0.0, 0.0, 0.0, 0.0

        # Class mapping
        class_idx = self.config.LABEL_MAPPING.get(label_str, 2)  # default to Stellar Variability / Unknown
        
        # Confidence score target: 1.0 for true planet transit, 0.0 for others
        confidence = 1.0 if class_idx == 0 else 0.0

        # 3. Preprocessing
        # Standard preprocessing pipeline
        processed_flux = LightCurvePreprocessor.preprocess(time, flux, config=self.config)
        
        # 4. Generate Option A (Full Curve) or Option B (BLS Window)
        if self.use_bls_windows:
            # Run BLS search to center the window around transit candidate
            bls_res = TransitCandidateDetector.run_bls(
                time, 
                processed_flux,
                min_period=0.5,
                max_period=min(20.0, max(1.0, period * 1.5)) if period > 0 else 20.0
            )
            # Crop window of size BLS_WINDOW_SIZE centered on transit mid-time
            input_flux = TransitCandidateDetector.extract_transit_window(
                time,
                processed_flux,
                bls_res['time0'],
                bls_res['period'],
                window_size=self.config.BLS_WINDOW_SIZE,
                align_method=self.config.ALIGN_METHOD
            )
            input_time = np.linspace(-0.5, 0.5, self.config.BLS_WINDOW_SIZE)
        else:
            # Symmetrically crop/pad to target length
            input_flux = LightCurvePreprocessor.align_sequence_length(
                processed_flux, 
                target_len=self.config.SEQUENCE_LENGTH,
                method=self.config.ALIGN_METHOD
            )
            input_time = np.linspace(0, 10, self.config.SEQUENCE_LENGTH)

        # 5. Apply Data Augmentations (Only for train dataset)
        if self.augment:
            input_flux = TimeSeriesAugmenter.apply_augmentations(input_flux)

        # Normalize the final sequence to Z-score again if augmented
        if self.augment or self.use_bls_windows:
            input_flux = LightCurvePreprocessor.normalize_flux(input_flux, method=self.config.NORM_METHOD)

        # 6. Format input channels (shape: [Channels, Length])
        if self.config.INPUT_DIM == 1:
            # Flux only
            x_tensor = torch.tensor(input_flux, dtype=torch.float32).unsqueeze(0)
        else:
            # Time and Flux
            # Normalize time sequence to 0-1 range to balance channels
            norm_time = (input_time - np.min(input_time)) / (np.max(input_time) - np.min(input_time) + 1e-8)
            x_tensor = torch.stack([
                torch.tensor(input_flux, dtype=torch.float32),
                torch.tensor(norm_time, dtype=torch.float32)
            ], dim=0)

        # Wrap targets in Tensors
        y_class = torch.tensor(class_idx, dtype=torch.long)
        
        # Log-scale or scale regression parameters to be in a comparable range for stability
        # Depth is in ppm, Period in days, Duration in hours, Midpoint in days
        # We can predict them raw and apply weights in the loss, or log-transform
        y_reg = torch.tensor([
            np.log1p(depth) / 10.0,        # Scale depth
            duration / 10.0,               # Scale duration
            np.log1p(period) / 5.0,        # Scale period
            np.log1p(abs(time0bk)) / 10.0  # Scale midpoint
        ], dtype=torch.float32)

        y_conf = torch.tensor(confidence, dtype=torch.float32).unsqueeze(0)

        return x_tensor, y_class, y_reg, y_conf


def get_data_loaders(config=Config, test_size=0.15, val_size=0.15, seed=42):
    """
    Loads indexes and metadata, performs a stratified train/val/test split,
    and returns PyTorch DataLoaders.
    """
    # Load dataset index csv
    if not os.path.exists(config.INDEX_PATH):
        raise FileNotFoundError(f"Index file {config.INDEX_PATH} not found.")
    df_index = pd.read_csv(config.INDEX_PATH)
    
    # Load metadata koi cumulative csv
    if not os.path.exists(config.METADATA_PATH):
        raise FileNotFoundError(f"Metadata file {config.METADATA_PATH} not found.")
    df_meta = pd.read_csv(config.METADATA_PATH, comment='#')

    # Keep only files that actually exist
    existing_records = []
    for _, row in df_index.iterrows():
        filepath = row["file"].replace("\\", "/")
        full_path = os.path.join(config.BASE_DIR, filepath)
        if os.path.exists(full_path):
            existing_records.append(row)
            
    if len(existing_records) == 0:
        raise ValueError("No valid light curve .npz files found under the specified paths.")
        
    df_valid = pd.DataFrame(existing_records)
    
    # Ensure label exists in LABEL_MAPPING, fallback to unknown (unlabeled)
    df_valid["label_idx"] = df_valid["label"].apply(lambda x: config.LABEL_MAPPING.get(x, 2))

    # Stratified split: Train_Val and Test
    train_val_df, test_df = train_test_split(
        df_valid, 
        test_size=test_size, 
        random_state=seed, 
        stratify=df_valid["label_idx"]
    )
    
    # Calculate relative validation split size
    rel_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=rel_val_size, 
        random_state=seed, 
        stratify=train_val_df["label_idx"]
    )

    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Instantiate datasets
    train_ds = ExoplanetDataset(train_df, df_meta, config=config, augment=True)
    val_ds = ExoplanetDataset(val_df, df_meta, config=config, augment=False)
    test_ds = ExoplanetDataset(test_df, df_meta, config=config, augment=False)

    # Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader
