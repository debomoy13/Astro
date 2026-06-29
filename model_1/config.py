import os

class Config:
    """
    Configuration parameters for the ISRO hackathon exoplanet detection,
    event classification, and parameter estimation pipeline.
    """
    # ------------------
    # Data Directories & Paths
    # ------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    
    # Path to csv files
    INDEX_PATH = os.path.join(BASE_DIR, "dataset_index.csv")
    if not os.path.exists(INDEX_PATH):
        INDEX_PATH = os.path.join(BASE_DIR, "modified datasets", "dataset_index.csv")
        
    METADATA_PATH = os.path.join(BASE_DIR, "koi_cumulative_labeled.csv")
    if not os.path.exists(METADATA_PATH):
        METADATA_PATH = os.path.join(BASE_DIR, "modified datasets", "koi_cumulative_labeled.csv")

    # Outputs
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "model_1", "checkpoints")
    RESULTS_DIR = os.path.join(BASE_DIR, "model_1", "results")

    # Ensure output directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------
    # Preprocessing
    # ------------------
    SEQUENCE_LENGTH = 2000          # Option A: Full light curve sequence length
    USE_BLS_WINDOWS = False          # Toggle between Option A (Full Curve) and Option B (BLS Windows)
    BLS_WINDOW_SIZE = 200           # Option B: Candidate window length
    ALIGN_METHOD = 'edge'           # 'edge', 'zero', 'symmetric' padding
    NORM_METHOD = 'zscore'          # 'zscore', 'minmax', 'median'
    
    # Detrending / Smoothing parameters
    SIGMA_CLIPPING_SIGMA = 3.0
    SIGMA_CLIPPING_ITERS = 2
    SG_WINDOW_SIZE = 15
    SG_POLYORDER = 2
    MEDIAN_FILTER_WINDOW = 101

    # ------------------
    # Model Architecture
    # ------------------
    INPUT_DIM = 1                   # 1: flux only, 2: [time, flux]
    NUM_CLASSES = 5                 # 5: Planet, Binary, Variability, Noise, Blend
    
    # Stage 1: Conv1D Extractor
    CNN_CHANNELS = [32, 64, 128]
    CNN_KERNEL_SIZES = [15, 11, 7]
    CNN_POOL_SIZE = 4
    CNN_USE_RESIDUAL = True
    
    # Stage 2: Temporal Attention
    ATTENTION_HEADS = 4
    ATTENTION_DIM = 128             # Must match final CNN output channel size
    
    # Stage 3: BiLSTM
    LSTM_HIDDEN_DIM = 128
    LSTM_LAYERS = 2
    
    # Stage 4: Shared Feature Embedding
    SHARED_EMBEDDING_DIM = 128
    DROPOUT = 0.3

    # ------------------
    # Loss Weights
    # ------------------
    # L_total = L_class + lambda_1*L_depth + lambda_2*L_dur + lambda_3*L_per + lambda_4*L_mid + lambda_5*L_conf
    LAMBDA_CLASSIFICATION = 1.0
    LAMBDA_DEPTH = 0.1
    LAMBDA_DURATION = 0.1
    LAMBDA_PERIOD = 0.1
    LAMBDA_MIDPOINT = 0.1
    LAMBDA_CONFIDENCE = 1.0

    # ------------------
    # Training Parameters
    # ------------------
    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIPPING = 5.0
    MIXED_PRECISION = True
    EARLY_STOPPING_PATIENCE = 8
    
    # Class mapping
    LABEL_MAPPING = {
        'transit': 0,           # Exoplanet Transit
        'stellar_eclipse': 1,   # Eclipsing Binary
        'not_transit': 2,       # Stellar Variability (mapped contextually) or Noise
        'centroid_offset': 3,   # Instrument Noise / Offset
        'Ephemeris match': 4,   # Blend / False Positive
        'unlabeled': 2,         # Mapping unlabeled/other to 2 (Variability/Unknown)
    }

    # Inverse mapping for display
    CLASS_NAMES = [
        "Exoplanet Transit",
        "Eclipsing Binary",
        "Stellar Variability",
        "Instrument Noise",
        "Blend/False Positive"
    ]
