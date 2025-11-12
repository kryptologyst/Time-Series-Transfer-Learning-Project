"""
Anomaly detection methods for time series data.

This module includes various anomaly detection algorithms including
statistical methods, machine learning approaches, and deep learning models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging
from dataclasses import dataclass
import warnings

# Statistical methods
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# PyOD for advanced anomaly detection
try:
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.lof import LOF
    from pyod.models.copod import COPOD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    warnings.warn("PyOD not available. Install with: pip install pyod")

logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection models."""
    contamination: float = 0.1
    threshold_method: str = "iqr"  # "iqr", "zscore", "modified_zscore", "isolation_forest"
    z_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    window_size: int = 10
    autoencoder_encoding_dim: int = 16
    autoencoder_learning_rate: float = 0.001
    autoencoder_epochs: int = 50
    random_state: int = 42


class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection."""
    
    def __init__(self, config: AnomalyConfig):
        """Initialize statistical anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.thresholds = {}
        
    def z_score_detection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Detect anomalies using Z-score method.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (anomaly_mask, thresholds)
        """
        mean = np.mean(data)
        std = np.std(data)
        
        z_scores = np.abs((data - mean) / std)
        anomaly_mask = z_scores > self.config.z_threshold
        
        thresholds = {
            'mean': mean,
            'std': std,
            'threshold': self.config.z_threshold
        }
        
        return anomaly_mask, thresholds
    
    def modified_z_score_detection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Detect anomalies using modified Z-score method.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (anomaly_mask, thresholds)
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = np.std(data)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        anomaly_mask = np.abs(modified_z_scores) > self.config.z_threshold
        
        thresholds = {
            'median': median,
            'mad': mad,
            'threshold': self.config.z_threshold
        }
        
        return anomaly_mask, thresholds
    
    def iqr_detection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (anomaly_mask, thresholds)
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.config.iqr_multiplier * iqr
        upper_bound = q3 + self.config.iqr_multiplier * iqr
        
        anomaly_mask = (data < lower_bound) | (data > upper_bound)
        
        thresholds = {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        return anomaly_mask, thresholds
    
    def rolling_window_detection(self, data: np.ndarray, 
                               method: str = "z_score") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using rolling window approach.
        
        Args:
            data: Time series data
            method: Detection method ("z_score", "iqr", "modified_z_score")
            
        Returns:
            Tuple of (anomaly_mask, window_stats)
        """
        window_size = self.config.window_size
        anomaly_mask = np.zeros(len(data), dtype=bool)
        window_stats = []
        
        for i in range(window_size, len(data) - window_size):
            window_data = data[i-window_size//2:i+window_size//2]
            
            if method == "z_score":
                mask, thresholds = self.z_score_detection(window_data)
            elif method == "iqr":
                mask, thresholds = self.iqr_detection(window_data)
            elif method == "modified_z_score":
                mask, thresholds = self.modified_z_score_detection(window_data)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Check if current point is anomalous in its window
            center_idx = window_size // 2
            if mask[center_idx]:
                anomaly_mask[i] = True
            
            window_stats.append(thresholds)
        
        return anomaly_mask, {'window_stats': window_stats}


class MLAnomalyDetector:
    """Machine learning methods for anomaly detection."""
    
    def __init__(self, config: AnomalyConfig):
        """Initialize ML anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        
    def isolation_forest_detection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using Isolation Forest.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (anomaly_mask, model_info)
        """
        # Reshape data for sklearn
        X = data.reshape(-1, 1)
        
        # Fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_state
        )
        
        anomaly_scores = self.model.fit_predict(X)
        anomaly_mask = anomaly_scores == -1
        
        model_info = {
            'contamination': self.config.contamination,
            'n_anomalies': np.sum(anomaly_mask),
            'anomaly_rate': np.mean(anomaly_mask)
        }
        
        return anomaly_mask, model_info
    
    def one_class_svm_detection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using One-Class SVM.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (anomaly_mask, model_info)
        """
        # Reshape data for sklearn
        X = data.reshape(-1, 1)
        
        # Fit One-Class SVM
        self.model = OneClassSVM(
            nu=self.config.contamination,
            kernel='rbf'
        )
        
        anomaly_scores = self.model.fit_predict(X)
        anomaly_mask = anomaly_scores == -1
        
        model_info = {
            'nu': self.config.contamination,
            'n_anomalies': np.sum(anomaly_mask),
            'anomaly_rate': np.mean(anomaly_mask)
        }
        
        return anomaly_mask, model_info
    
    def dbscan_detection(self, data: np.ndarray, 
                        eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using DBSCAN clustering.
        
        Args:
            data: Time series data
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Tuple of (anomaly_mask, model_info)
        """
        # Reshape data for sklearn
        X = data.reshape(-1, 1)
        
        # Fit DBSCAN
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.model.fit_predict(X)
        
        # Points labeled as -1 are considered anomalies
        anomaly_mask = cluster_labels == -1
        
        model_info = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_anomalies': np.sum(anomaly_mask),
            'anomaly_rate': np.mean(anomaly_mask)
        }
        
        return anomaly_mask, model_info


class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detection."""
    
    def __init__(self, config: AnomalyConfig):
        """Initialize autoencoder anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        
    def create_autoencoder(self, input_dim: int) -> nn.Module:
        """Create autoencoder model.
        
        Args:
            input_dim: Input dimension
            
        Returns:
            Autoencoder model
        """
        class AutoEncoder(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, encoding_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return AutoEncoder(input_dim, self.config.autoencoder_encoding_dim)
    
    def create_sequences(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Create sequences for autoencoder training.
        
        Args:
            data: Time series data
            window_size: Window size for sequences
            
        Returns:
            Array of sequences
        """
        sequences = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i + window_size])
        return np.array(sequences)
    
    def fit(self, data: np.ndarray, window_size: int = 10) -> None:
        """Fit autoencoder model.
        
        Args:
            data: Time series data
            window_size: Window size for sequences
        """
        logger.info("Training autoencoder for anomaly detection")
        
        # Create sequences
        sequences = self.create_sequences(data, window_size)
        
        # Normalize data
        sequences_scaled = self.scaler.fit_transform(sequences.reshape(-1, window_size))
        sequences_scaled = sequences_scaled.reshape(-1, window_size)
        
        # Create model
        self.model = self.create_autoencoder(window_size)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=self.config.autoencoder_learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(sequences_scaled)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.autoencoder_epochs):
            total_loss = 0
            for batch in dataloader:
                batch_data = batch[0]
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_data)
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.5f}")
    
    def detect_anomalies(self, data: np.ndarray, window_size: int = 10,
                        threshold_percentile: float = 95) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using trained autoencoder.
        
        Args:
            data: Time series data
            window_size: Window size for sequences
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Tuple of (anomaly_mask, model_info)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before detection")
        
        # Create sequences
        sequences = self.create_sequences(data, window_size)
        
        # Normalize data
        sequences_scaled = self.scaler.transform(sequences.reshape(-1, window_size))
        sequences_scaled = sequences_scaled.reshape(-1, window_size)
        
        # Get reconstructions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(sequences_scaled)
            reconstructions = self.model(X_tensor)
            
            # Calculate reconstruction errors
            reconstruction_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).numpy()
        
        # Determine threshold
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        # Create anomaly mask for full data
        anomaly_mask = np.zeros(len(data), dtype=bool)
        for i, error in enumerate(reconstruction_errors):
            if error > threshold:
                # Mark the entire window as anomalous
                start_idx = i
                end_idx = min(i + window_size, len(data))
                anomaly_mask[start_idx:end_idx] = True
        
        model_info = {
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'n_anomalies': np.sum(anomaly_mask),
            'anomaly_rate': np.mean(anomaly_mask),
            'mean_reconstruction_error': np.mean(reconstruction_errors)
        }
        
        return anomaly_mask, model_info


class PyODAnomalyDetector:
    """PyOD-based anomaly detection methods."""
    
    def __init__(self, config: AnomalyConfig):
        """Initialize PyOD anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is not available. Install with: pip install pyod")
        
        self.config = config
        self.model = None
        
    def lof_detection(self, data: np.ndarray, n_neighbors: int = 20) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using Local Outlier Factor (LOF).
        
        Args:
            data: Time series data
            n_neighbors: Number of neighbors for LOF
            
        Returns:
            Tuple of (anomaly_mask, model_info)
        """
        # Reshape data for PyOD
        X = data.reshape(-1, 1)
        
        # Fit LOF model
        self.model = LOF(contamination=self.config.contamination, n_neighbors=n_neighbors)
        self.model.fit(X)
        
        # Get anomaly scores
        anomaly_scores = self.model.decision_scores_
        anomaly_mask = self.model.labels_ == 1
        
        model_info = {
            'n_neighbors': n_neighbors,
            'contamination': self.config.contamination,
            'n_anomalies': np.sum(anomaly_mask),
            'anomaly_rate': np.mean(anomaly_mask),
            'mean_anomaly_score': np.mean(anomaly_scores)
        }
        
        return anomaly_mask, model_info
    
    def copod_detection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using COPOD (Copula-based Outlier Detection).
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (anomaly_mask, model_info)
        """
        # Reshape data for PyOD
        X = data.reshape(-1, 1)
        
        # Fit COPOD model
        self.model = COPOD(contamination=self.config.contamination)
        self.model.fit(X)
        
        # Get anomaly scores
        anomaly_scores = self.model.decision_scores_
        anomaly_mask = self.model.labels_ == 1
        
        model_info = {
            'contamination': self.config.contamination,
            'n_anomalies': np.sum(anomaly_mask),
            'anomaly_rate': np.mean(anomaly_mask),
            'mean_anomaly_score': np.mean(anomaly_scores)
        }
        
        return anomaly_mask, model_info


class EnsembleAnomalyDetector:
    """Ensemble anomaly detection combining multiple methods."""
    
    def __init__(self, config: AnomalyConfig):
        """Initialize ensemble anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.detectors = {}
        self.weights = {}
        
    def add_detector(self, name: str, detector: Any, weight: float = 1.0) -> None:
        """Add a detector to the ensemble.
        
        Args:
            name: Detector name
            detector: Anomaly detector instance
            weight: Detector weight
        """
        self.detectors[name] = detector
        self.weights[name] = weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
    
    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using ensemble of detectors.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (anomaly_mask, ensemble_info)
        """
        detector_results = {}
        
        # Run all detectors
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'detect_anomalies'):
                    anomaly_mask, info = detector.detect_anomalies(data)
                elif hasattr(detector, 'isolation_forest_detection'):
                    anomaly_mask, info = detector.isolation_forest_detection(data)
                elif hasattr(detector, 'z_score_detection'):
                    anomaly_mask, info = detector.z_score_detection(data)
                else:
                    logger.warning(f"Unknown detector method for {name}")
                    continue
                
                detector_results[name] = {
                    'anomaly_mask': anomaly_mask,
                    'info': info,
                    'weight': self.weights[name]
                }
                
            except Exception as e:
                logger.warning(f"Detector {name} failed: {e}")
        
        if not detector_results:
            raise ValueError("No detectors successfully completed")
        
        # Combine results using weighted voting
        ensemble_mask = np.zeros(len(data), dtype=bool)
        ensemble_scores = np.zeros(len(data))
        
        for name, result in detector_results.items():
            weight = result['weight']
            mask = result['anomaly_mask']
            
            # Add weighted contribution
            ensemble_scores += weight * mask.astype(float)
            
            # Majority voting
            ensemble_mask |= mask
        
        # Final decision based on threshold
        threshold = 0.5  # Majority threshold
        final_mask = ensemble_scores >= threshold
        
        ensemble_info = {
            'detector_results': detector_results,
            'ensemble_scores': ensemble_scores,
            'threshold': threshold,
            'n_anomalies': np.sum(final_mask),
            'anomaly_rate': np.mean(final_mask)
        }
        
        return final_mask, ensemble_info


if __name__ == "__main__":
    # Example usage
    config = AnomalyConfig()
    
    # Generate sample data with anomalies
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    data = np.sin(t) + 0.1 * np.random.randn(200)
    
    # Add some anomalies
    data[50] += 3
    data[100] -= 2.5
    data[150] += 4
    
    # Test statistical detector
    stat_detector = StatisticalAnomalyDetector(config)
    anomaly_mask, thresholds = stat_detector.z_score_detection(data)
    
    print(f"Statistical detection found {np.sum(anomaly_mask)} anomalies")
    print(f"Anomaly rate: {np.mean(anomaly_mask):.3f}")
    
    # Test ML detector
    ml_detector = MLAnomalyDetector(config)
    anomaly_mask_ml, info = ml_detector.isolation_forest_detection(data)
    
    print(f"ML detection found {np.sum(anomaly_mask_ml)} anomalies")
    print(f"Anomaly rate: {np.mean(anomaly_mask_ml):.3f}")
