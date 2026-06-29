import torch
from model_1.config import Config
from model_1.model import ExoplanetDeepModel

def test_dimensions():
    print("=" * 60)
    print("SHAPE VERIFICATION SUITE")
    print("=" * 60)
    
    # Instantiate config and model
    config = Config()
    model = ExoplanetDeepModel(config=config)
    model.eval()
    
    # ------------------
    # Option A: Full Light Curve
    # ------------------
    batch_size = 4
    seq_len_a = config.SEQUENCE_LENGTH # 2000
    
    # Mock input: (B, Channels, Length) -> (4, 1, 2000)
    dummy_input_a = torch.randn(batch_size, config.INPUT_DIM, seq_len_a)
    print(f"Option A - Input Shape:               {dummy_input_a.shape}")
    
    # Run forward pass
    class_logits, reg_outputs, confidence, attn_weights = model(dummy_input_a)
    
    print(f"Option A - Classification Logits:     {class_logits.shape}  (Expected: [4, 5])")
    print(f"Option A - Regression Outputs:         {reg_outputs.shape}  (Expected: [4, 4])")
    print(f"Option A - Confidence Score:           {confidence.shape}  (Expected: [4, 1])")
    print(f"Option A - Attention Weights:          {attn_weights.shape}  (Expected: [4, 500, 500])")
    
    # Assertions for Option A
    assert class_logits.shape == (batch_size, config.NUM_CLASSES)
    assert reg_outputs.shape == (batch_size, 4)
    assert confidence.shape == (batch_size, 1)
    
    # ------------------
    # Option B: Candidate Transit Window
    # ------------------
    seq_len_b = config.BLS_WINDOW_SIZE # 200
    
    # Mock input: (B, Channels, Length) -> (4, 1, 200)
    dummy_input_b = torch.randn(batch_size, config.INPUT_DIM, seq_len_b)
    print(f"\nOption B - Input Shape:               {dummy_input_b.shape}")
    
    # Run forward pass
    class_logits_b, reg_outputs_b, confidence_b, attn_weights_b = model(dummy_input_b)
    
    print(f"Option B - Classification Logits:     {class_logits_b.shape}  (Expected: [4, 5])")
    print(f"Option B - Regression Outputs:         {reg_outputs_b.shape}  (Expected: [4, 4])")
    print(f"Option B - Confidence Score:           {confidence_b.shape}  (Expected: [4, 1])")
    print(f"Option B - Attention Weights:          {attn_weights_b.shape}  (Expected: [4, 50, 50])")
    
    # Assertions for Option B
    assert class_logits_b.shape == (batch_size, config.NUM_CLASSES)
    assert reg_outputs_b.shape == (batch_size, 4)
    assert confidence_b.shape == (batch_size, 1)
    
    print("=" * 60)
    print("STATUS: ALL DIMENSIONALITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_dimensions()
