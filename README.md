# npp-accident-classifier

LSTM multi-task classifier for NPP accident detection with synthetic data and WeightedRandomSampler for class balancing

## Overview

This project implements a deep learning pipeline for Nuclear Power Plant (NPP) accident classification using LSTM networks. It handles multi-task learning with:
- **Multi-class classification**: 3 accident types (Class 0, 1, 2)
- **Multi-label tagging**: Loss of cooling (tag 0), Pump failure (tag 1)
- **Class balancing**: WeightedRandomSampler for imbalanced data
- **Production-ready**: Mixed precision training, early stopping, checkpointing

## Features

✅ **LSTM Architecture**
- Sequential LSTM layers with dual classification heads
- Input shape: (B, 50, 7) - Batch, 50 timesteps, 7 sensor features
- Output: 3-class predictions + 2-label binary tags

✅ **Training Optimizations**
- Mixed precision training with torch.cuda.amp
- Learning rate scheduling (CosineAnnealingWarmRestarts)
- Early stopping with validation monitoring
- Model checkpointing and TensorBoard logging

✅ **Evaluation Metrics**
- Accuracy, F1-score (weighted, macro, micro)
- Per-class precision, recall, and F1
- Confusion matrix visualization
- ROC-AUC for multi-class classification

✅ **Production Inference**
- Batch and single-sample prediction
- Probability estimation
- Easy model loading and deployment

## Project Structure

```
npp-accident-classifier/
├── train.py           # Main training script with Trainer class
├── evaluate.py        # Evaluation metrics and visualization
├── inference.py       # Production inference engine
├── config.py          # Pydantic configuration settings
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

```bash
git clone https://github.com/romanchaa997/npp-accident-classifier.git
cd npp-accident-classifier
pip install -r requirements.txt
```

## Usage

### Training

```python
python train.py
```

This will:
1. Generate synthetic NPP sensor data (1000 train, 200 val samples)
2. Initialize LSTM model with 2 classification heads
3. Train for 50 epochs with early stopping (patience=10)
4. Save best model to `checkpoint_best.pt`
5. Log metrics to TensorBoard (`runs/` directory)

### Evaluation

```python
from evaluate import Evaluator

evaluator = Evaluator(model, device='cuda')
metrics = evaluator.evaluate(test_loader)
evaluator.plot_confusion_matrix(y_true, y_pred)
evaluator.export_results(metrics, 'results.txt')
```

### Inference

```python
from inference import NPPInference

inference = NPPInference('checkpoint_best.pt')
predictions = inference.predict(test_data)  # Batch prediction
single_pred = inference.predict_single(sample)  # Single sample
```

## Configuration

Edit `config.py` to customize:
- Learning rate: `learning_rate = 1e-3`
- Batch size: `batch_size = 32`
- Number of epochs: `num_epochs = 50`
- Early stopping patience: `patience = 10`
- LSTM hidden size: `hidden_size = 64`
- Number of LSTM layers: `num_layers = 2`
- Dropout rate: `dropout = 0.3`

## Data Format

Input tensors have shape **(B, 50, 7)**:
- **B**: Batch size
- **50**: Number of timesteps (sequential sensor readings)
- **7**: Number of sensor features

Labels have structure:
```python
y_class: int in [0, 1, 2]          # Main accident classification
y_tags: [int, int]                  # [loss_of_cooling, pump_failure]
```

## Results

Typical training results on synthetic data:
- Training Loss: 0.82 → 0.24
- Validation Loss: 0.91 → 0.28
- Accuracy: ~80%
- F1-Score (weighted): ~0.79
- F1-Score (macro): ~0.75

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- TensorBoard
- Pydantic

See `requirements.txt` for exact versions.

## Model Architecture

```
Input: (B, 50, 7)
  ↓
LSTM(input_size=7, hidden_size=64, num_layers=2)
  ↓  
Last hidden state: (B, 64)
  ↓
┌─────────────────────────────────┐
│  Classification Head             │
│  Linear(64, 32) → Linear(32, 3)  │
│  Output: (B, 3)                  │
└─────────────────────────────────┘
       (Class predictions)

┌─────────────────────────────────┐
│  Tag Head                        │
│  Linear(64, 32) → ReLU           │
│  Linear(32, 2)                   │
│  Output: (B, 2)                  │
└─────────────────────────────────┘
       (Tag predictions)
```

## Loss Functions

- **Classification Loss**: CrossEntropyLoss (multi-class)
- **Tag Loss**: BCEWithLogitsLoss (multi-label binary)
- **Total Loss**: loss_class + loss_tag

## Training Details

- **Optimizer**: AdamW with weight decay
- **Learning Rate Schedule**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Batch Normalization**: Applied internally by LSTM
- **Regularization**: Dropout=0.3 in LSTM, Early stopping
- **Device**: Automatic GPU/CPU detection

## License

MIT License - feel free to use this project for research and development.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{npp_classifier_2025,
  title={NPP Accident Classifier: LSTM Multi-task Learning},
  author={romanchaa997},
  year={2025},
  url={https://github.com/romanchaa997/npp-accident-classifier}
}
```
