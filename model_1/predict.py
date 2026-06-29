import os
import torch
import numpy as np
import argparse

from model_1.config import Config
from model_1.preprocessing import LightCurvePreprocessor
from model_1.candidate_detection import TransitCandidateDetector
from model_1.model import ExoplanetDeepModel
from model_1.utils import plot_attention_heatmap

class ExoplanetPredictor:
    """
    Executes single-sample inference on Kepler light curves using the trained model checkpoint.
    """
    def __init__(self, checkpoint_path=None, config=Config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # Load the best checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            
        self.model = ExoplanetDeepModel(config=config)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded trained model weights from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running with uninitialized weights.")
            
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_numpy(self, time, flux):
        """
        Predicts classes, regression parameters, and confidence from raw time and flux arrays.
        """
        # 1. Preprocess
        processed_flux = LightCurvePreprocessor.preprocess(time, flux, config=self.config)
        
        # 2. Sequence Extraction (Option A or Option B)
        if self.config.USE_BLS_WINDOWS:
            bls_res = TransitCandidateDetector.run_bls(time, processed_flux)
            input_flux = TransitCandidateDetector.extract_transit_window(
                time, processed_flux,
                bls_res['time0'], bls_res['period'],
                window_size=self.config.BLS_WINDOW_SIZE,
                align_method=self.config.ALIGN_METHOD
            )
            input_time = np.linspace(-0.5, 0.5, self.config.BLS_WINDOW_SIZE)
        else:
            input_flux = LightCurvePreprocessor.align_sequence_length(
                processed_flux, 
                target_len=self.config.SEQUENCE_LENGTH,
                method=self.config.ALIGN_METHOD
            )
            input_time = np.linspace(0, 10, self.config.SEQUENCE_LENGTH)

        # Re-normalize to Z-score
        input_flux = LightCurvePreprocessor.normalize_flux(input_flux, method=self.config.NORM_METHOD)

        # 3. Format tensor
        if self.config.INPUT_DIM == 1:
            x_tensor = torch.tensor(input_flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            norm_time = (input_time - np.min(input_time)) / (np.max(input_time) - np.min(input_time) + 1e-8)
            x_tensor = torch.stack([
                torch.tensor(input_flux, dtype=torch.float32),
                torch.tensor(norm_time, dtype=torch.float32)
            ], dim=0).unsqueeze(0)
            
        x_tensor = x_tensor.to(self.device)

        # 4. Forward pass
        class_logits, reg_outputs, confidence, attn_weights = self.model(x_tensor)
        
        # 5. Extract output outputs
        # Classification probabilities
        probs = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        class_name = self.config.CLASS_NAMES[pred_class]
        
        # Confidence score
        conf_score = float(confidence.cpu().numpy()[0, 0])
        
        # Scale back regression outputs to raw physical dimensions
        reg = reg_outputs.cpu().numpy()[0]
        pred_depth = float(np.expm1(reg[0] * 10.0))       # ppm
        pred_duration = float(reg[1] * 10.0)              # hours
        pred_period = float(np.expm1(reg[2] * 5.0))       # days
        pred_midpoint = float(np.expm1(reg[3] * 10.0))     # time0bk days
        
        # Attention scores (average head attention weights)
        attn_weights = attn_weights.cpu().numpy()[0]
        
        results = {
            'predicted_class': pred_class,
            'predicted_class_name': class_name,
            'probabilities': {self.config.CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
            'confidence_score': conf_score,
            'transit_parameters': {
                'depth_ppm': pred_depth,
                'duration_hours': pred_duration,
                'orbital_period_days': pred_period,
                'transit_epoch_midpoint': pred_midpoint
            },
            'attention_weights': attn_weights,
            'preprocessed_flux': input_flux
        }
        
        return results

    def predict_file(self, npz_path):
        """
        Loads .npz light curve file and executes prediction.
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Light curve file {npz_path} not found.")
            
        data = np.load(npz_path)
        if "time" not in data or "flux" not in data:
            raise KeyError("NPZ file must contain keys 'time' and 'flux'.")
            
        return self.predict_numpy(data["time"], data["flux"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exoplanet Pipeline Inference")
    parser.add_argument("--file", type=str, required=True, help="Path to light curve .npz file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint .pt file")
    parser.add_argument("--save_attention", action="store_true", help="Save temporal attention heatmap overlay plot")
    
    args = parser.parse_args()
    
    predictor = ExoplanetPredictor(checkpoint_path=args.checkpoint)
    res = predictor.predict_file(args.file)
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Classification:      {res['predicted_class_name']}")
    print(f"Confidence Score:    {res['confidence_score']:.2%}")
    print("\nClass Probabilities:")
    for k, v in res['probabilities'].items():
        print(f"  - {k}: {v:.2%}")
        
    print("\nEstimated Transit Parameters:")
    for k, v in res['transit_parameters'].items():
        print(f"  - {k.replace('_', ' ').title()}: {v:.4f}")
    print("="*50)
    
    if args.save_attention:
        out_path = os.path.join(Config.RESULTS_DIR, "prediction_attention_map.png")
        plot_attention_heatmap(
            res['preprocessed_flux'], 
            res['attention_weights'], 
            output_dir=Config.RESULTS_DIR,
            filename="prediction_attention_map.png"
        )
        print(f"Saved temporal attention map overlay plot to {out_path}")
