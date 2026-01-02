"""Production inference script for NPP accident classifier."""

import torch
import numpy as np

class NPPInference:
    """Inference engine for NPP accident classification."""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        print(f'Inference engine initialized on {self.device}')
    
    def predict(self, x):
        """Make predictions on input data.
        
        Args:
            x: Input tensor (B, T, F) - B=batch, T=timesteps, F=features
        
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None:
            raise RuntimeError('Model not loaded')
        
        x = torch.from_numpy(x).float().to(self.device) if isinstance(x, np.ndarray) else x.to(self.device)
        
        with torch.no_grad():
            class_logits, tag_logits = self.model(x)
            class_probs = torch.softmax(class_logits, dim=1)
            class_preds = torch.argmax(class_logits, dim=1)
            tag_probs = torch.sigmoid(tag_logits)
            tag_preds = (tag_probs > 0.5).int()
        
        return {
            'class_pred': class_preds.cpu().numpy(),
            'class_prob': class_probs.cpu().numpy(),
            'tag_pred': tag_preds.cpu().numpy(),
            'tag_prob': tag_probs.cpu().numpy()
        }
    
    def predict_single(self, x):
        """Single sample prediction."""
        x = np.expand_dims(x, axis=0)
        results = self.predict(x)
        return {
            'class': int(results['class_pred'][0]),
            'class_prob': float(results['class_prob'][0].max()),
            'tags': [int(t) for t in results['tag_pred'][0]]
        }
