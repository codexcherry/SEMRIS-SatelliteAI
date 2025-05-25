import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import xarray as xr

class EnvironmentalRNN(nn.Module):
    """RNN-based model for environmental parameter prediction."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 dropout: float = 0.2):
        """
        Initialize the RNN model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of RNN layers
            output_size (int): Number of output features
            dropout (float): Dropout rate
        """
        super(EnvironmentalRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            hidden (Tuple[torch.Tensor, torch.Tensor]): Initial hidden state
            
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output and hidden state
        """
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # LSTM forward pass
        out, hidden = self.lstm(x, hidden)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out, hidden

class ModelTrainer:
    """Handles training and prediction for the environmental RNN model."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 10000,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize the model trainer.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of RNN layers
            output_size (int): Number of output features
            dropout (float): Dropout rate
            learning_rate (float): Learning rate for optimization
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate input size based on spatial dimensions (100x100 grid)
        self.spatial_size = 100 * 100  # n_lats * n_lons
        
        self.model = EnvironmentalRNN(
            input_size=self.spatial_size,  # Use flattened spatial dimensions
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def prepare_data(self,
                    dataset: xr.Dataset,
                    sequence_length: int,
                    target_parameter: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data with validation checks and debugging outputs.
        
        Args:
            dataset: Input dataset
            sequence_length: Length of input sequences
            target_parameter: Parameter to predict
            
        Returns:
            Tuple of (input_sequences, targets)
        """
        data = dataset[target_parameter].values
        
        # Data validation
        print(f"\n=== Data Validation ===")
        print(f"Raw data shape: {data.shape}")
        print(f"Data stats - Min: {np.min(data):.2f}, Max: {np.max(data):.2f}, Mean: {np.mean(data):.2f}")
        print(f"NaN values: {np.isnan(data).sum()}, Zero values: {(data == 0).sum()}")
        
        if np.all(data == 0):
            raise ValueError("Received all-zero data from API - check data source")
        if np.isnan(data).any():
            print("Warning: NaN values detected, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0)
        
        n_times, n_lats, n_lons = data.shape
        data_reshaped = data.reshape(n_times, -1)  # Flatten spatial dimensions
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_reshaped) - sequence_length):
            X.append(data_reshaped[i:i + sequence_length])
            y.append(data_reshaped[i + sequence_length])
        
        # Convert to tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        print(f"\n=== Training Data Prepared ===")
        print(f"Input sequences shape: {X.shape}")
        print(f"Targets shape: {y.shape}")
        print(f"First input sample stats - Min: {X[0].min():.2f}, Max: {X[0].max():.2f}, Mean: {X[0].mean():.2f}")
        print(f"First target sample stats - Min: {y[0].min():.2f}, Max: {y[0].max():.2f}, Mean: {y[0].mean():.2f}")
        
        return X, y
    
    def train(self,
              X: torch.Tensor,
              y: torch.Tensor,
              epochs: int,
              batch_size: int = 32) -> List[float]:
        """
        Train the model with enhanced monitoring.
        
        Args:
            X: Input sequences
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            List of training losses
        """
        if X.shape[0] == 0:
            raise ValueError("No training data available - check sequence_length parameter")
            
        print(f"\n=== Training Started ===")
        print(f"Training samples: {len(X)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]
            
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                output, _ = self.model(batch_X)
                last_output = output[:, -1, :]
                loss = self.criterion(last_output, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss for epoch
            epoch_loss /= len(X) / batch_size
            losses.append(epoch_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs-1:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
                # Print sample predictions
                with torch.no_grad():
                    sample_pred = self.model(X[:1])[0][:,-1,:]
                    print(f"Sample prediction stats - Min: {sample_pred.min():.2f}, Max: {sample_pred.max():.2f}, Mean: {sample_pred.mean():.2f}")
        
        print("\n=== Training Completed ===")
        return losses
    
    def predict(self,
               X: torch.Tensor,
               sequence_length: int) -> np.ndarray:
        """
        Make predictions with debugging outputs.
        
        Args:
            X: Input sequences
            sequence_length: Prediction horizon
            
        Returns:
            Predictions array
        """
        print(f"\n=== Prediction Started ===")
        print(f"Input shape: {X.shape}")
        print(f"Prediction horizon: {sequence_length}")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            current_input = X[-1:]  # Use last sequence
            hidden = None
            
            for i in range(sequence_length):
                output, hidden = self.model(current_input, hidden)
                last_output = output[:, -1, :]
                predictions.append(last_output.cpu().numpy())
                current_input = last_output.unsqueeze(1)
                
                if i % 10 == 0 or i == sequence_length-1:
                    print(f"Step {i+1}/{sequence_length} - Prediction stats: Min: {last_output.min():.2f}, Max: {last_output.max():.2f}, Mean: {last_output.mean():.2f}")
        
        predictions = np.vstack(predictions)
        predictions = predictions.reshape(sequence_length, 100, 100)
        
        print("\n=== Prediction Results ===")
        print(f"Final predictions shape: {predictions.shape}")
        print(f"Final stats - Min: {np.min(predictions):.2f}, Max: {np.max(predictions):.2f}, Mean: {np.mean(predictions):.2f}")
        
        return predictions
    
    def save_model(self, path: str):
        """
        Save the trained model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """
        Load a trained model.
        
        Args:
            path (str): Path to the saved model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 