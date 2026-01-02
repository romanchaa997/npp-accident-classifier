"""NPP Accident Classification - Training Script

Full training pipeline with mixed precision, gradient accumulation,
learning rate scheduling, early stopping, and model checkpointing.
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


class NPPModel(nn.Module):
    """LSTM-based multi-task classifier for NPP accident detection."""
    
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.3, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.tag_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        class_logits = self.fc2(self.relu(self.fc1(last_out)))
        tag_logits = self.tag_head(last_out)
        
        return class_logits, tag_logits


class Trainer:
    """Training manager with mixed precision and early stopping."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 1e-3))
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_tag = nn.BCEWithLogitsLoss()
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x, y_class, y_tags) in enumerate(tqdm(self.train_loader)):
            x = x.to(self.device)
            y_class = y_class.long().to(self.device)
            y_tags = y_tags.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                class_logits, tag_logits = self.model(x)
                loss_class = self.criterion_class(class_logits, y_class)
                loss_tag = self.criterion_tag(tag_logits, y_tags)
                loss = loss_class + loss_tag
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), 
                                      self.epoch * len(self.train_loader) + batch_idx)
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y_class, y_tags in tqdm(self.val_loader):
                x = x.to(self.device)
                y_class = y_class.long().to(self.device)
                y_tags = y_tags.float().to(self.device)
                
                class_logits, tag_logits = self.model(x)
                loss_class = self.criterion_class(class_logits, y_class)
                loss_tag = self.criterion_tag(tag_logits, y_tags)
                loss = loss_class + loss_tag
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Val/Loss', avg_loss, self.epoch)
        return avg_loss
    
    def train(self, num_epochs=100, patience=10):
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {train_loss:.4f} - '
                  f'Val Loss: {val_loss:.4f}')
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(f'checkpoint_best.pt')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': self.epoch
        }, path)


def create_synthetic_data(num_samples=1000):
    """Generate synthetic NPP sensor data."""
    # (B, T=50, F=7) - B batches, 50 timesteps, 7 features
    X = np.random.randn(num_samples, 50, 7).astype(np.float32)
    # Multi-class labels (0, 1, 2)
    y_class = np.random.randint(0, 3, num_samples).astype(np.int64)
    # Multi-label tags
    y_tags = np.random.randint(0, 2, (num_samples, 2)).astype(np.float32)
    return X, y_class, y_tags


if __name__ == '__main__':
    # Configuration
    config = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_epochs': 50,
        'patience': 10,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3
    }
    
    # Create synthetic data
    print('Generating synthetic data...')
    X_train, y_train_class, y_train_tags = create_synthetic_data(1000)
    X_val, y_val_class, y_val_tags = create_synthetic_data(200)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train_class),
        torch.from_numpy(y_train_tags)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val_class),
        torch.from_numpy(y_val_tags)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model and trainer
    model = NPPModel(
        input_size=7,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_classes=3
    )
    
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train model
    print('Starting training...')
    trainer.train(num_epochs=config['num_epochs'], patience=config['patience'])
    print('Training complete!')
