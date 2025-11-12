# Time Series Transfer Learning Project

A comprehensive Python project for time series analysis using transfer learning techniques, featuring forecasting, anomaly detection, and interactive visualization.

## Features

- **Transfer Learning**: Train models on source data and fine-tune on target data
- **Multiple Forecasting Methods**: ARIMA, Prophet, LSTM, GRU, and Transformer models
- **Anomaly Detection**: Statistical and machine learning approaches including Isolation Forest, Autoencoders
- **Interactive Visualization**: Streamlit web interface and Plotly charts
- **Comprehensive Testing**: Unit tests for all components
- **Modern Architecture**: Type hints, docstrings, configuration management

## Project Structure

```
├── src/                          # Source code
│   ├── data_generation.py        # Synthetic data generation
│   ├── models.py                 # Deep learning models (LSTM, GRU, Transformer)
│   ├── forecasting.py            # Traditional forecasting methods
│   ├── anomaly_detection.py     # Anomaly detection algorithms
│   ├── visualization.py          # Plotting and visualization
│   ├── main.py                  # Main application
│   └── streamlit_app.py         # Web interface
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── data/                         # Data storage
├── models/                       # Saved models
├── plots/                        # Generated plots
├── logs/                         # Log files
├── tests/                        # Unit tests
│   └── test_all.py              # Comprehensive test suite
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Time-Series-Transfer-Learning-Project.git
   cd Time-Series-Transfer-Learning-Project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**:
   ```bash
   mkdir -p data models plots logs notebooks
   ```

## Usage

### Command Line Interface

#### Complete Analysis
Run the full pipeline with default settings:
```bash
python src/main.py --mode complete
```

#### Individual Components
```bash
# Generate data only
python src/main.py --mode data

# Train model only
python src/main.py --mode train

# Run forecasting only
python src/main.py --mode forecast

# Run anomaly detection only
python src/main.py --mode anomaly

# Create visualizations only
python src/main.py --mode visualize
```

#### Custom Configuration
```bash
python src/main.py --config custom_config.yaml --mode complete
```

### Web Interface (Streamlit)

Launch the interactive web interface:
```bash
streamlit run src/streamlit_app.py
```

The web interface provides:
- Real-time parameter adjustment
- Interactive visualizations
- Model comparison dashboards
- Anomaly detection results
- Training progress monitoring

### Python API

#### Basic Usage
```python
from src.main import TimeSeriesAnalysisApp

# Initialize application
app = TimeSeriesAnalysisApp("config/config.yaml")

# Run complete analysis
results = app.run_complete_analysis()

# Access results
datasets = results['datasets']
forecasts = results['forecasts']
anomaly_results = results['anomaly_results']
```

#### Custom Data Generation
```python
from src.data_generation import TimeSeriesConfig, TimeSeriesGenerator

# Configure data generation
config = TimeSeriesConfig(
    length=500,
    noise_level=0.1,
    phase_shift=np.pi/4,
    random_seed=42
)

# Generate data
generator = TimeSeriesGenerator(config)
series = generator.generate_sine_wave()
```

#### Transfer Learning
```python
from src.models import ModelConfig, TransferLearningTrainer, create_model

# Configure model
model_config = ModelConfig(
    hidden_dim=32,
    learning_rate=0.005,
    epochs_pretrain=10,
    epochs_finetune=5
)

# Create and train model
model = create_model('lstm', model_config)
trainer = TransferLearningTrainer(model, model_config)

# Pretrain on source data
trainer.pretrain(X_source, y_source)

# Fine-tune on target data
trainer.finetune(X_target, y_target)
```

#### Forecasting
```python
from src.forecasting import ARIMAForecaster, ProphetForecaster

# ARIMA forecasting
arima_forecaster = ARIMAForecaster(ForecastConfig())
arima_forecaster.fit(data, auto_order=True)
forecast, lower, upper = arima_forecaster.forecast(30)

# Prophet forecasting
prophet_forecaster = ProphetForecaster(ForecastConfig())
prophet_forecaster.fit(data)
forecast, lower, upper = prophet_forecaster.forecast(30)
```

#### Anomaly Detection
```python
from src.anomaly_detection import StatisticalAnomalyDetector, MLAnomalyDetector

# Statistical methods
stat_detector = StatisticalAnomalyDetector(AnomalyConfig())
anomaly_mask, info = stat_detector.z_score_detection(data)

# Machine learning methods
ml_detector = MLAnomalyDetector(AnomalyConfig())
anomaly_mask, info = ml_detector.isolation_forest_detection(data)
```

## Configuration

The project uses YAML configuration files. Key configuration options:

### Data Generation
```yaml
data:
  source:
    length: 500
    noise: 0.1
    shift: 0.0
  target:
    length: 500
    noise: 0.1
    shift: 0.785  # π/4
  sequence_length: 30
```

### Model Configuration
```yaml
models:
  lstm:
    hidden_dim: 32
    num_layers: 1
    dropout: 0.1
    learning_rate: 0.005
    batch_size: 32
    epochs_pretrain: 10
    epochs_finetune: 5
```

### Forecasting
```yaml
models:
  arima:
    order: [1, 1, 1]
    seasonal_order: [1, 1, 1, 12]
  prophet:
    yearly_seasonality: true
    weekly_seasonality: true
    daily_seasonality: false
```

### Anomaly Detection
```yaml
anomaly_detection:
  isolation_forest:
    contamination: 0.1
    random_state: 42
  autoencoder:
    encoding_dim: 16
    learning_rate: 0.001
    epochs: 50
```

## Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/ -v
```

Or run specific test modules:
```bash
python tests/test_all.py
```

The test suite covers:
- Data generation and preprocessing
- Model training and evaluation
- Forecasting accuracy
- Anomaly detection performance
- Visualization functionality
- Error handling and edge cases

## Dependencies

### Core Dependencies
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.7.0` - Static plotting
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.15.0` - Interactive plotting

### Time Series Analysis
- `statsmodels>=0.14.0` - Statistical models
- `pmdarima>=2.0.0` - Auto ARIMA
- `prophet>=1.1.0` - Facebook Prophet
- `tslearn>=0.6.0` - Time series machine learning
- `darts>=0.24.0` - Time series forecasting
- `sktime>=0.21.0` - Scikit-learn for time series

### Deep Learning
- `torch>=2.0.0` - PyTorch framework
- `torchvision>=0.15.0` - Computer vision
- `transformers>=4.30.0` - Transformer models

### Machine Learning
- `scikit-learn>=1.3.0` - Machine learning
- `xgboost>=1.7.0` - Gradient boosting
- `lightgbm>=4.0.0` - Light gradient boosting
- `pyod>=1.1.0` - Outlier detection

### Visualization and UI
- `streamlit>=1.25.0` - Web interface
- `jupyter>=1.0.0` - Jupyter notebooks
- `ipywidgets>=8.0.0` - Interactive widgets

### Utilities
- `pyyaml>=6.0` - YAML configuration
- `python-dotenv>=1.0.0` - Environment variables
- `tqdm>=4.65.0` - Progress bars
- `joblib>=1.3.0` - Parallel processing

### Development
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage testing
- `black>=23.0.0` - Code formatting
- `flake8>=6.0.0` - Linting
- `mypy>=1.5.0` - Type checking

## Examples

### Example 1: Basic Transfer Learning
```python
from src.main import TimeSeriesAnalysisApp

# Initialize and run analysis
app = TimeSeriesAnalysisApp()
results = app.run_complete_analysis()

# Access transfer learning results
trainer = results['trainer']
print(f"Final training loss: {trainer.history['finetune_loss'][-1]:.5f}")
```

### Example 2: Custom Forecasting Pipeline
```python
from src.forecasting import ARIMAForecaster, ProphetForecaster, EnsembleForecaster

# Generate sample data
import numpy as np
data = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.randn(100)

# Individual models
arima_forecaster = ARIMAForecaster(ForecastConfig())
arima_forecaster.fit(data)
arima_forecast, _, _ = arima_forecaster.forecast(20)

prophet_forecaster = ProphetForecaster(ForecastConfig())
prophet_forecaster.fit(data)
prophet_forecast, _, _ = prophet_forecaster.forecast(20)

# Ensemble
ensemble_forecaster = EnsembleForecaster(ForecastConfig())
ensemble_forecaster.add_model('ARIMA', arima_forecaster, weight=0.5)
ensemble_forecaster.add_model('Prophet', prophet_forecaster, weight=0.5)
ensemble_forecaster.fit(data)
ensemble_forecast, _, _ = ensemble_forecaster.forecast(20)
```

### Example 3: Anomaly Detection Comparison
```python
from src.anomaly_detection import StatisticalAnomalyDetector, MLAnomalyDetector

# Generate data with anomalies
data = np.random.randn(100)
data[20] += 5  # Add anomaly
data[60] -= 4  # Add anomaly

# Compare methods
stat_detector = StatisticalAnomalyDetector(AnomalyConfig())
ml_detector = MLAnomalyDetector(AnomalyConfig())

z_score_mask, _ = stat_detector.z_score_detection(data)
iqr_mask, _ = stat_detector.iqr_detection(data)
isolation_mask, _ = ml_detector.isolation_forest_detection(data)

print(f"Z-Score detected: {np.sum(z_score_mask)} anomalies")
print(f"IQR detected: {np.sum(iqr_mask)} anomalies")
print(f"Isolation Forest detected: {np.sum(isolation_mask)} anomalies")
```

## Performance Considerations

- **GPU Acceleration**: PyTorch models automatically use GPU if available
- **Memory Management**: Large datasets are processed in batches
- **Parallel Processing**: Multiple models can be trained simultaneously
- **Caching**: Model checkpoints and results are saved for reuse

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce batch size or sequence length in configuration

3. **CUDA Errors**: Install appropriate PyTorch version for your system
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Prophet Installation**: On some systems, Prophet requires additional dependencies
   ```bash
   conda install -c conda-forge prophet
   ```

### Logging

Enable debug logging by modifying the configuration:
```yaml
logging:
  level: "DEBUG"
  file: "logs/debug.log"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
python -m pytest tests/ -v

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{time_series_transfer_learning,
  title={Time Series Transfer Learning Project},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Time-Series-Transfer-Learning-Project}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Statsmodels contributors for statistical methods
- Facebook Research for Prophet forecasting
- Streamlit team for the web interface framework
- The open-source community for various libraries and tools

---

**Note**: This project is designed for educational and research purposes. Always validate results and consider domain-specific requirements when applying to real-world problems.
# Time-Series-Transfer-Learning-Project
