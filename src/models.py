"""
Deep learning models for time series transfer learning.

This module contains LSTM, GRU, and Transformer models with transfer learning capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 0.005
    batch_size: int = 32
    epochs_pretrain: int = 10
    epochs_finetune: int = 5
    sequence_length: int = 30
    device: str = "auto"


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting with transfer learning capabilities."""
    
    def __init__(self, input_size: int = 1, hidden_dim: int = 32, 
                 num_layers: int = 1, dropout: float = 0.1, 
                 output_size: int = 1):
        """Initialize LSTM model.
        
        Args:
            input_size: Size of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Size of output
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = hidden[-1]  # Take the last layer's hidden state
        
        # Apply dropout and fully connected layer
        output = self.dropout(last_hidden)
        output = self.fc(output)
        
        return output.squeeze(-1) if output.size(-1) == 1 else output


class GRUModel(nn.Module):
    """GRU model for time series forecasting."""
    
    def __init__(self, input_size: int = 1, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.1,
                 output_size: int = 1):
        """Initialize GRU model.
        
        Args:
            input_size: Size of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            output_size: Size of output
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        gru_out, hidden = self.gru(x)
        last_hidden = hidden[-1]
        
        output = self.dropout(last_hidden)
        output = self.fc(output)
        
        return output.squeeze(-1) if output.size(-1) == 1 else output


class TransformerModel(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(self, input_size: int = 1, d_model: int = 64, 
                 nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, output_size: int = 1):
        """Initialize Transformer model.
        
        Args:
            input_size: Size of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            output_size: Size of output
        """
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Use the last timestep for prediction
        last_output = transformer_out[:, -1, :]
        
        # Project to output size
        output = self.output_projection(last_output)
        
        return output.squeeze(-1) if output.size(-1) == 1 else output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransferLearningTrainer:
    """Trainer class for transfer learning with time series models."""
    
    def __init__(self, model: nn.Module, config: ModelConfig):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            config: Model configuration
        """
        self.model = model
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training history
        self.history = {
            'pretrain_loss': [],
            'finetune_loss': [],
            'val_loss': []
        }
        
    def create_data_loader(self, X: np.ndarray, y: np.ndarray, 
                          shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader.
        
        Args:
            X: Input sequences
            y: Target values
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader object
        """
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)
    
    def pretrain(self, X_source: np.ndarray, y_source: np.ndarray) -> None:
        """Pretrain model on source domain.
        
        Args:
            X_source: Source input sequences
            y_source: Source target values
        """
        logger.info("Starting pretraining on source domain")
        
        data_loader = self.create_data_loader(X_source, y_source)
        
        self.model.train()
        for epoch in range(self.config.epochs_pretrain):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.history['pretrain_loss'].append(avg_loss)
            
            if epoch % 2 == 0:
                logger.info(f"Pretrain Epoch {epoch+1}/{self.config.epochs_pretrain}, Loss: {avg_loss:.5f}")
    
    def finetune(self, X_target: np.ndarray, y_target: np.ndarray,
                 X_val: Optional[np.ndarray] = None, 
                 y_val: Optional[np.ndarray] = None) -> None:
        """Fine-tune model on target domain.
        
        Args:
            X_target: Target input sequences
            y_target: Target values
            X_val: Validation input sequences
            y_val: Validation target values
        """
        logger.info("Starting fine-tuning on target domain")
        
        train_loader = self.create_data_loader(X_target, y_target)
        val_loader = None
        
        if X_val is not None and y_val is not None:
            val_loader = self.create_data_loader(X_val, y_val, shuffle=False)
        
        self.model.train()
        for epoch in range(self.config.epochs_finetune):
            # Training
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.history['finetune_loss'].append(avg_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                logger.info(f"Finetune Epoch {epoch+1}/{self.config.epochs_finetune}, "
                           f"Train Loss: {avg_loss:.5f}, Val Loss: {val_loss:.5f}")
            else:
                logger.info(f"Finetune Epoch {epoch+1}/{self.config.epochs_finetune}, Loss: {avg_loss:.5f}")
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on given data.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Model loaded from {filepath}")


def create_model(model_type: str, config: ModelConfig) -> nn.Module:
    """Factory function to create models.
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'transformer')
        config: Model configuration
        
    Returns:
        Initialized model
    """
    if model_type.lower() == 'lstm':
        return LSTMModel(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    elif model_type.lower() == 'gru':
        return GRUModel(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    elif model_type.lower() == 'transformer':
        return TransformerModel(
            d_model=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    config = ModelConfig(hidden_dim=32, epochs_pretrain=5, epochs_finetune=3)
    
    # Create model
    model = create_model('lstm', config)
    trainer = TransferLearningTrainer(model, config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using device: {trainer.device}")
