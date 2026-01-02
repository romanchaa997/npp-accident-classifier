"""Evaluation script for NPP accident classifier model."""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, confusion_matrix, roc_auc_score,
    classification_report, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """Model evaluation and visualization utilities."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, test_loader):
        """Evaluate model on test dataset."""
        all_preds_class = []
        all_targets_class = []
        all_preds_tag = []
        all_targets_tag = []
        
        with torch.no_grad():
            for x, y_class, y_tags in test_loader:
                x = x.to(self.device)
                y_class = y_class.to(self.device)
                y_tags = y_tags.to(self.device)
                
                class_logits, tag_logits = self.model(x)
                
                class_preds = torch.argmax(class_logits, dim=1).cpu().numpy()
                all_preds_class.extend(class_preds)
                all_targets_class.extend(y_class.cpu().numpy())
                
                tag_preds = (torch.sigmoid(tag_logits) > 0.5).int().cpu().numpy()
                all_preds_tag.extend(tag_preds)
                all_targets_tag.extend(y_tags.int().cpu().numpy())
        
        all_preds_class = np.array(all_preds_class)
        all_targets_class = np.array(all_targets_class)
        all_preds_tag = np.array(all_preds_tag)
        all_targets_tag = np.array(all_targets_tag)
        
        return self.compute_metrics(all_targets_class, all_preds_class,
                                   all_targets_tag, all_preds_tag)
    
    def compute_metrics(self, y_true_class, y_pred_class, y_true_tag, y_pred_tag):
        """Compute classification metrics."""
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true_class, y_pred_class)
        metrics['f1_weighted'] = f1_score(y_true_class, y_pred_class, average='weighted')
        metrics['f1_macro'] = f1_score(y_true_class, y_pred_class, average='macro')
        metrics['f1_micro'] = f1_score(y_true_class, y_pred_class, average='micro')
        
        metrics['tag_f1'] = []
        for i in range(y_true_tag.shape[1]):
            f1 = f1_score(y_true_tag[:, i], y_pred_tag[:, i])
            metrics['tag_f1'].append(f1)
        
        metrics['confusion_matrix'] = confusion_matrix(y_true_class, y_pred_class)
        metrics['report'] = classification_report(y_true_class, y_pred_class)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f'Saved confusion matrix to {save_path}')
        plt.close()
