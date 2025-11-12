#!/usr/bin/env python3
"""
Setup script for Time Series Transfer Learning Project.

This script helps users set up the project environment and run initial tests.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    """Print setup header."""
    print("=" * 60)
    print("TIME SERIES TRANSFER LEARNING PROJECT SETUP")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python {version.major}.{version.minor} detected.")
        print("   This project requires Python 3.10 or higher.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected.")
        return True


def create_directories():
    """Create necessary directories."""
    print("\nCreating project directories...")
    
    directories = [
        "data",
        "models", 
        "plots",
        "logs",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    print("✅ All directories created successfully.")


def install_dependencies():
    """Install project dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print("✅ pip upgraded successfully.")
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ All dependencies installed successfully.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("   Please check your internet connection and try again.")
        return False


def run_tests():
    """Run project tests."""
    print("\nRunning tests...")
    
    try:
        # Run tests
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed successfully!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ pytest not found. Installing pytest...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest"], 
                          check=True, capture_output=True)
            print("✅ pytest installed. Please run tests manually:")
            print("   python -m pytest tests/ -v")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install pytest.")
            return False


def create_sample_config():
    """Create a sample configuration file."""
    print("\nCreating sample configuration...")
    
    config_content = """# Sample configuration for Time Series Transfer Learning Project
data:
  source:
    type: "synthetic"
    function: "sine_wave"
    shift: 0
    noise: 0.1
    length: 500
    frequency: 1.0
  
  target:
    type: "synthetic"
    function: "sine_wave"
    shift: 0.785  # π/4
    noise: 0.1
    length: 500
    frequency: 1.0
  
  sequence_length: 30
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

models:
  lstm:
    hidden_dim: 32
    num_layers: 1
    dropout: 0.1
    learning_rate: 0.005
    batch_size: 32
    epochs_pretrain: 10
    epochs_finetune: 5
  
  arima:
    order: [1, 1, 1]
    seasonal_order: [1, 1, 1, 12]
  
  prophet:
    yearly_seasonality: true
    weekly_seasonality: true
    daily_seasonality: false
    seasonality_mode: "multiplicative"

anomaly_detection:
  isolation_forest:
    contamination: 0.1
    random_state: 42
  
  autoencoder:
    encoding_dim: 16
    learning_rate: 0.001
    epochs: 50

visualization:
  figure_size: [12, 8]
  style: "seaborn-v0_8"
  color_palette: "husl"
  save_plots: true
  plot_format: "png"
  dpi: 300

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/timeseries.log"

paths:
  data_dir: "data"
  models_dir: "models"
  plots_dir: "plots"
  logs_dir: "logs"
"""
    
    config_path = Path("config/sample_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Sample configuration created: {config_path}")


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("\n1. Run the complete analysis:")
    print("   python src/main.py --mode complete")
    print("\n2. Launch the web interface:")
    print("   streamlit run src/streamlit_app.py")
    print("\n3. Run individual components:")
    print("   python src/main.py --mode data      # Generate data only")
    print("   python src/main.py --mode train     # Train model only")
    print("   python src/main.py --mode forecast  # Run forecasting only")
    print("   python src/main.py --mode anomaly   # Run anomaly detection only")
    print("\n4. Run tests:")
    print("   python -m pytest tests/ -v")
    print("\n5. Explore the Jupyter notebook:")
    print("   jupyter notebook notebooks/demo_analysis.ipynb")
    print("\n6. Customize configuration:")
    print("   Edit config/config.yaml for your specific needs")
    print("\nFor more information, see README.md")
    print("\nHappy analyzing! 🎉")


def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependency installation failed.")
        print("   You can try installing manually:")
        print("   pip install -r requirements.txt")
        print("\n   Continuing with setup...")
    
    # Create sample config
    create_sample_config()
    
    # Run tests (optional)
    print("\nWould you like to run tests now? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            run_tests()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    
    # Print usage instructions
    print_usage_instructions()


if __name__ == "__main__":
    main()
