#!/usr/bin/env python3
"""
Quick demo script for Time Series Transfer Learning Project.

This script demonstrates the basic functionality without requiring
all dependencies to be installed.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import our modules
try:
    from src.data_generation import TimeSeriesConfig, TimeSeriesGenerator
    from src.models import ModelConfig, create_model
    from src.anomaly_detection import AnomalyConfig, StatisticalAnomalyDetector
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run: pip install -r requirements.txt")
    exit(1)


def demo_data_generation():
    """Demonstrate data generation capabilities."""
    print("\n" + "="*50)
    print("DEMO: Data Generation")
    print("="*50)
    
    # Create configuration
    config = TimeSeriesConfig(
        length=200,
        noise_level=0.1,
        phase_shift=np.pi/4,
        random_seed=42
    )
    
    # Generate data
    generator = TimeSeriesGenerator(config)
    series = generator.generate_sine_wave()
    
    print(f"Generated time series with {len(series)} points")
    print(f"Mean: {np.mean(series):.3f}")
    print(f"Std: {np.std(series):.3f}")
    print(f"Min: {np.min(series):.3f}")
    print(f"Max: {np.max(series):.3f}")
    
    # Plot the data
    plt.figure(figsize=(10, 4))
    plt.plot(series, linewidth=2)
    plt.title("Generated Sine Wave Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return series


def demo_anomaly_detection(data):
    """Demonstrate anomaly detection capabilities."""
    print("\n" + "="*50)
    print("DEMO: Anomaly Detection")
    print("="*50)
    
    # Add some anomalies
    data_with_anomalies = data.copy()
    data_with_anomalies[50] += 3
    data_with_anomalies[100] -= 2.5
    data_with_anomalies[150] += 4
    
    # Configure anomaly detection
    config = AnomalyConfig(
        contamination=0.1,
        z_threshold=3.0,
        random_state=42
    )
    
    # Run different detection methods
    detector = StatisticalAnomalyDetector(config)
    
    # Z-score detection
    z_score_mask, z_info = detector.z_score_detection(data_with_anomalies)
    print(f"Z-Score method detected {np.sum(z_score_mask)} anomalies")
    
    # IQR detection
    iqr_mask, iqr_info = detector.iqr_detection(data_with_anomalies)
    print(f"IQR method detected {np.sum(iqr_mask)} anomalies")
    
    # Modified Z-score detection
    mod_z_mask, mod_z_info = detector.modified_z_score_detection(data_with_anomalies)
    print(f"Modified Z-Score method detected {np.sum(mod_z_mask)} anomalies")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original data
    axes[0, 0].plot(data, 'b-', linewidth=2)
    axes[0, 0].set_title("Original Data")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Data with anomalies
    axes[0, 1].plot(data_with_anomalies, 'r-', linewidth=2)
    axes[0, 1].scatter([50, 100, 150], data_with_anomalies[[50, 100, 150]], 
                     color='red', s=100, zorder=5)
    axes[0, 1].set_title("Data with Anomalies")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z-score detection
    normal_indices = ~z_score_mask
    axes[1, 0].plot(np.where(normal_indices)[0], data_with_anomalies[normal_indices], 
                   'b-', linewidth=1, alpha=0.7)
    anomaly_indices = np.where(z_score_mask)[0]
    if len(anomaly_indices) > 0:
        axes[1, 0].scatter(anomaly_indices, data_with_anomalies[anomaly_indices], 
                          color='red', s=50, zorder=5)
    axes[1, 0].set_title(f"Z-Score Detection ({np.sum(z_score_mask)} anomalies)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # IQR detection
    normal_indices = ~iqr_mask
    axes[1, 1].plot(np.where(normal_indices)[0], data_with_anomalies[normal_indices], 
                   'b-', linewidth=1, alpha=0.7)
    anomaly_indices = np.where(iqr_mask)[0]
    if len(anomaly_indices) > 0:
        axes[1, 1].scatter(anomaly_indices, data_with_anomalies[anomaly_indices], 
                          color='red', s=50, zorder=5)
    axes[1, 1].set_title(f"IQR Detection ({np.sum(iqr_mask)} anomalies)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demo_model_creation():
    """Demonstrate model creation capabilities."""
    print("\n" + "="*50)
    print("DEMO: Model Creation")
    print("="*50)
    
    # Create model configuration
    config = ModelConfig(
        hidden_dim=16,
        num_layers=1,
        dropout=0.1,
        learning_rate=0.01,
        batch_size=16,
        epochs_pretrain=2,
        epochs_finetune=1
    )
    
    # Create different model types
    try:
        lstm_model = create_model('lstm', config)
        print(f"✅ LSTM model created with {sum(p.numel() for p in lstm_model.parameters())} parameters")
        
        gru_model = create_model('gru', config)
        print(f"✅ GRU model created with {sum(p.numel() for p in gru_model.parameters())} parameters")
        
        transformer_model = create_model('transformer', config)
        print(f"✅ Transformer model created with {sum(p.numel() for p in transformer_model.parameters())} parameters")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        print("This might be due to missing PyTorch installation.")


def demo_forecasting():
    """Demonstrate basic forecasting capabilities."""
    print("\n" + "="*50)
    print("DEMO: Basic Forecasting")
    print("="*50)
    
    # Generate sample data
    t = np.linspace(0, 4*np.pi, 100)
    data = np.sin(t) + 0.1 * np.random.randn(100)
    
    # Simple moving average forecast
    window_size = 10
    forecast = []
    
    for i in range(window_size, len(data)):
        forecast.append(np.mean(data[i-window_size:i]))
    
    forecast = np.array(forecast)
    
    print(f"Generated forecast with {len(forecast)} points")
    print(f"Forecast mean: {np.mean(forecast):.3f}")
    print(f"Forecast std: {np.std(forecast):.3f}")
    
    # Plot forecast
    plt.figure(figsize=(10, 4))
    plt.plot(data, 'b-', linewidth=2, label='Actual')
    plt.plot(range(window_size, len(data)), forecast, 'r--', linewidth=2, label='Forecast')
    plt.title("Simple Moving Average Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Main demo function."""
    print("TIME SERIES TRANSFER LEARNING - QUICK DEMO")
    print("=" * 60)
    
    try:
        # Run demos
        data = demo_data_generation()
        demo_anomaly_detection(data)
        demo_model_creation()
        demo_forecasting()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nTo run the full analysis:")
        print("1. Install all dependencies: pip install -r requirements.txt")
        print("2. Run complete analysis: python src/main.py --mode complete")
        print("3. Launch web interface: streamlit run src/streamlit_app.py")
        print("\nFor more information, see README.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()
