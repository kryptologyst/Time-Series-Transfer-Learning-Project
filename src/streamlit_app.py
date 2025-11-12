"""
Streamlit web interface for Time Series Transfer Learning Project.

This module provides an interactive web interface for exploring
time series analysis results and running experiments.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import logging

# Import our modules
from src.data_generation import TimeSeriesConfig, TimeSeriesGenerator, generate_transfer_learning_datasets
from src.models import ModelConfig, TransferLearningTrainer, create_model
from src.forecasting import ForecastConfig, ARIMAForecaster, ProphetForecaster, EnsembleForecaster
from src.anomaly_detection import AnomalyConfig, StatisticalAnomalyDetector, MLAnomalyDetector
from src.visualization import PlotConfig, TimeSeriesVisualizer, InteractiveVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Transfer Learning",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitApp:
    """Streamlit application class."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self._setup_session_state()
    
    def _setup_session_state(self):
        """Setup Streamlit session state variables."""
        if 'datasets' not in st.session_state:
            st.session_state.datasets = None
        if 'forecasts' not in st.session_state:
            st.session_state.forecasts = None
        if 'anomaly_results' not in st.session_state:
            st.session_state.anomaly_results = None
        if 'trainer' not in st.session_state:
            st.session_state.trainer = None
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">📈 Time Series Transfer Learning Analysis</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        This application demonstrates transfer learning techniques for time series analysis,
        including forecasting, anomaly detection, and model comparison.
        """)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("Configuration")
        
        # Data generation parameters
        st.sidebar.subheader("Data Generation")
        data_length = st.sidebar.slider("Data Length", 100, 1000, 500)
        noise_level = st.sidebar.slider("Noise Level", 0.01, 0.5, 0.1)
        phase_shift = st.sidebar.slider("Phase Shift (Target)", 0.0, 2*np.pi, np.pi/4)
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        hidden_dim = st.sidebar.selectbox("Hidden Dimension", [16, 32, 64, 128], index=1)
        learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.01, 0.005)
        epochs_pretrain = st.sidebar.slider("Pretrain Epochs", 5, 20, 10)
        epochs_finetune = st.sidebar.slider("Finetune Epochs", 3, 15, 5)
        
        # Forecasting parameters
        st.sidebar.subheader("Forecasting")
        forecast_horizon = st.sidebar.slider("Forecast Horizon", 10, 100, 30)
        
        # Anomaly detection parameters
        st.sidebar.subheader("Anomaly Detection")
        contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.2, 0.1)
        z_threshold = st.sidebar.slider("Z-Score Threshold", 2.0, 5.0, 3.0)
        
        return {
            'data_length': data_length,
            'noise_level': noise_level,
            'phase_shift': phase_shift,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'epochs_pretrain': epochs_pretrain,
            'epochs_finetune': epochs_finetune,
            'forecast_horizon': forecast_horizon,
            'contamination': contamination,
            'z_threshold': z_threshold
        }
    
    def generate_data(self, config):
        """Generate synthetic time series data."""
        with st.spinner("Generating synthetic time series data..."):
            # Create configurations
            source_config = TimeSeriesConfig(
                length=config['data_length'],
                noise_level=config['noise_level'],
                phase_shift=0.0,
                random_seed=42
            )
            
            target_config = TimeSeriesConfig(
                length=config['data_length'],
                noise_level=config['noise_level'],
                phase_shift=config['phase_shift'],
                random_seed=42
            )
            
            # Generate datasets
            datasets = generate_transfer_learning_datasets(source_config, target_config)
            st.session_state.datasets = datasets
            
            st.success("Data generation completed!")
            return datasets
    
    def train_model(self, config, datasets):
        """Train transfer learning model."""
        with st.spinner("Training transfer learning model..."):
            # Create model configuration
            model_config = ModelConfig(
                hidden_dim=config['hidden_dim'],
                learning_rate=config['learning_rate'],
                epochs_pretrain=config['epochs_pretrain'],
                epochs_finetune=config['epochs_finetune'],
                sequence_length=30
            )
            
            # Create and train model
            model = create_model('lstm', model_config)
            trainer = TransferLearningTrainer(model, model_config)
            
            # Pretrain on source data
            trainer.pretrain(datasets['source']['X'], datasets['source']['y'])
            
            # Fine-tune on target data
            trainer.finetune(
                datasets['target']['train']['X'], 
                datasets['target']['train']['y'],
                datasets['target']['val']['X'],
                datasets['target']['val']['y']
            )
            
            st.session_state.trainer = trainer
            st.success("Model training completed!")
            return trainer
    
    def run_forecasting(self, config, data):
        """Run forecasting analysis."""
        with st.spinner("Running forecasting analysis..."):
            forecasts = {}
            
            # ARIMA forecasting
            try:
                forecast_config = ForecastConfig(forecast_horizon=config['forecast_horizon'])
                arima_forecaster = ARIMAForecaster(forecast_config)
                arima_forecaster.fit(data, auto_order=True)
                forecast, lower, upper = arima_forecaster.forecast(config['forecast_horizon'])
                forecasts['ARIMA'] = (forecast, lower, upper)
            except Exception as e:
                st.warning(f"ARIMA forecasting failed: {e}")
            
            # Prophet forecasting (if available)
            try:
                prophet_forecaster = ProphetForecaster(forecast_config)
                prophet_forecaster.fit(data)
                forecast, lower, upper = prophet_forecaster.forecast(config['forecast_horizon'])
                forecasts['Prophet'] = (forecast, lower, upper)
            except Exception as e:
                st.warning(f"Prophet forecasting failed: {e}")
            
            st.session_state.forecasts = forecasts
            st.success("Forecasting analysis completed!")
            return forecasts
    
    def run_anomaly_detection(self, config, data):
        """Run anomaly detection analysis."""
        with st.spinner("Running anomaly detection analysis..."):
            anomaly_results = {}
            
            # Statistical methods
            anomaly_config = AnomalyConfig(
                contamination=config['contamination'],
                z_threshold=config['z_threshold']
            )
            
            stat_detector = StatisticalAnomalyDetector(anomaly_config)
            
            # Z-score detection
            anomaly_mask, info = stat_detector.z_score_detection(data)
            anomaly_results['Z-Score'] = (anomaly_mask, info)
            
            # IQR detection
            anomaly_mask, info = stat_detector.iqr_detection(data)
            anomaly_results['IQR'] = (anomaly_mask, info)
            
            # Machine learning methods
            ml_detector = MLAnomalyDetector(anomaly_config)
            
            # Isolation Forest
            anomaly_mask, info = ml_detector.isolation_forest_detection(data)
            anomaly_results['Isolation Forest'] = (anomaly_mask, info)
            
            st.session_state.anomaly_results = anomaly_results
            st.success("Anomaly detection analysis completed!")
            return anomaly_results
    
    def render_data_overview(self, datasets):
        """Render data overview section."""
        st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Source Data")
            source_data = datasets['source']['data']
            st.line_chart(source_data)
            
            # Statistics
            st.metric("Length", len(source_data))
            st.metric("Mean", f"{np.mean(source_data):.3f}")
            st.metric("Std", f"{np.std(source_data):.3f}")
        
        with col2:
            st.subheader("Target Data")
            target_data = datasets['target']['data']
            st.line_chart(target_data)
            
            # Statistics
            st.metric("Length", len(target_data))
            st.metric("Mean", f"{np.mean(target_data):.3f}")
            st.metric("Std", f"{np.std(target_data):.3f}")
        
        # Combined plot
        st.subheader("Source vs Target Comparison")
        combined_data = pd.DataFrame({
            'Source': source_data,
            'Target': target_data
        })
        st.line_chart(combined_data)
    
    def render_forecasting_results(self, data, forecasts):
        """Render forecasting results."""
        st.markdown('<h2 class="section-header">🔮 Forecasting Results</h2>', unsafe_allow_html=True)
        
        if not forecasts:
            st.warning("No forecasting results available. Please run the forecasting analysis.")
            return
        
        # Individual model forecasts
        for model_name, (forecast, lower, upper) in forecasts.items():
            st.subheader(f"{model_name} Forecast")
            
            # Create interactive plot
            fig = go.Figure()
            
            # Actual data
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            forecast_start = len(data) - len(forecast)
            forecast_x = list(range(forecast_start, forecast_start + len(forecast)))
            
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=list(upper) + list(lower[::-1]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"{model_name} Forecast",
                xaxis_title='Time',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        if len(forecasts) > 1:
            st.subheader("Model Comparison")
            
            fig = go.Figure()
            
            # Actual data
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ))
            
            # Forecasts
            colors = px.colors.qualitative.Set1
            for i, (model_name, (forecast, _, _)) in enumerate(forecasts.items()):
                forecast_start = len(data) - len(forecast)
                forecast_x = list(range(forecast_start, forecast_start + len(forecast)))
                
                fig.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast,
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
            
            fig.update_layout(
                title="Model Comparison",
                xaxis_title='Time',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_anomaly_detection_results(self, data, anomaly_results):
        """Render anomaly detection results."""
        st.markdown('<h2 class="section-header">🚨 Anomaly Detection Results</h2>', unsafe_allow_html=True)
        
        if not anomaly_results:
            st.warning("No anomaly detection results available. Please run the anomaly detection analysis.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        methods = list(anomaly_results.keys())
        anomaly_counts = [np.sum(mask) for mask, _ in anomaly_results.values()]
        anomaly_rates = [np.mean(mask) for mask, _ in anomaly_results.values()]
        
        with col1:
            st.metric("Methods Used", len(methods))
        with col2:
            st.metric("Avg Anomalies", f"{np.mean(anomaly_counts):.1f}")
        with col3:
            st.metric("Avg Anomaly Rate", f"{np.mean(anomaly_rates):.3f}")
        with col4:
            st.metric("Max Anomalies", max(anomaly_counts))
        
        # Individual method results
        for method_name, (anomaly_mask, info) in anomaly_results.items():
            st.subheader(f"{method_name} Results")
            
            # Create interactive plot
            fig = go.Figure()
            
            # Normal data
            normal_indices = ~anomaly_mask
            fig.add_trace(go.Scatter(
                x=np.where(normal_indices)[0],
                y=data[normal_indices],
                mode='lines',
                name='Normal',
                line=dict(color='blue', width=2)
            ))
            
            # Anomalies
            anomaly_indices = np.where(anomaly_mask)[0]
            if len(anomaly_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_indices,
                    y=data[anomaly_mask],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=8)
                ))
            
            fig.update_layout(
                title=f"{method_name} Anomaly Detection",
                xaxis_title='Time',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Method statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Detected", np.sum(anomaly_mask))
            with col2:
                st.metric("Anomaly Rate", f"{np.mean(anomaly_mask):.3f}")
            with col3:
                st.metric("Method Info", str(info)[:50] + "..." if len(str(info)) > 50 else str(info))
    
    def render_training_results(self, trainer):
        """Render training results."""
        st.markdown('<h2 class="section-header">🎯 Training Results</h2>', unsafe_allow_html=True)
        
        if trainer is None:
            st.warning("No training results available. Please train a model first.")
            return
        
        # Training history
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pretraining Loss")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=trainer.history['pretrain_loss'],
                mode='lines+markers',
                name='Pretrain Loss',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Pretraining Loss",
                xaxis_title='Epoch',
                yaxis_title='Loss'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Fine-tuning Loss")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=trainer.history['finetune_loss'],
                mode='lines+markers',
                name='Finetune Loss',
                line=dict(color='red', width=2)
            ))
            
            if trainer.history['val_loss']:
                fig.add_trace(go.Scatter(
                    y=trainer.history['val_loss'],
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='green', width=2)
                ))
            
            fig.update_layout(
                title="Fine-tuning Loss",
                xaxis_title='Epoch',
                yaxis_title='Loss'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Pretrain Loss", f"{trainer.history['pretrain_loss'][-1]:.5f}")
        with col2:
            st.metric("Final Finetune Loss", f"{trainer.history['finetune_loss'][-1]:.5f}")
        with col3:
            if trainer.history['val_loss']:
                st.metric("Final Validation Loss", f"{trainer.history['val_loss'][-1]:.5f}")
            else:
                st.metric("Validation Loss", "N/A")
    
    def run(self):
        """Run the Streamlit application."""
        # Render header
        self.render_header()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Home", "📊 Data", "🔮 Forecasting", "🚨 Anomaly Detection", "🎯 Training"
        ])
        
        with tab1:
            st.markdown("""
            ## Welcome to Time Series Transfer Learning Analysis
            
            This application demonstrates various time series analysis techniques including:
            
            - **Transfer Learning**: Train models on source data and fine-tune on target data
            - **Forecasting**: Multiple forecasting methods including ARIMA and Prophet
            - **Anomaly Detection**: Statistical and machine learning approaches
            - **Visualization**: Interactive plots and comprehensive analysis
            
            ### Getting Started
            
            1. Configure parameters in the sidebar
            2. Navigate through the tabs to explore different analyses
            3. Use the "Run Analysis" buttons to generate results
            
            ### Features
            
            - **Real-time Configuration**: Adjust parameters and see immediate results
            - **Interactive Visualizations**: Zoom, pan, and explore plots
            - **Multiple Models**: Compare different forecasting and anomaly detection methods
            - **Comprehensive Analysis**: From data generation to model evaluation
            """)
            
            # Quick start buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Run Complete Analysis", type="primary"):
                    datasets = self.generate_data(config)
                    trainer = self.train_model(config, datasets)
                    forecasts = self.run_forecasting(config, datasets['target']['data'])
                    anomaly_results = self.run_anomaly_detection(config, datasets['target']['data'])
                    st.success("Complete analysis finished!")
            
            with col2:
                if st.button("📊 Generate Data Only"):
                    datasets = self.generate_data(config)
            
            with col3:
                if st.button("🔄 Reset All"):
                    for key in ['datasets', 'forecasts', 'anomaly_results', 'trainer']:
                        st.session_state[key] = None
                    st.success("All data reset!")
        
        with tab2:
            if st.session_state.datasets is not None:
                self.render_data_overview(st.session_state.datasets)
            else:
                st.info("No data available. Please generate data first.")
                
                if st.button("Generate Data"):
                    datasets = self.generate_data(config)
        
        with tab3:
            if st.session_state.datasets is not None and st.session_state.forecasts is not None:
                self.render_forecasting_results(st.session_state.datasets['target']['data'], 
                                              st.session_state.forecasts)
            else:
                st.info("No forecasting results available. Please run forecasting analysis first.")
                
                if st.button("Run Forecasting Analysis"):
                    if st.session_state.datasets is not None:
                        forecasts = self.run_forecasting(config, st.session_state.datasets['target']['data'])
                    else:
                        st.warning("Please generate data first.")
        
        with tab4:
            if st.session_state.datasets is not None and st.session_state.anomaly_results is not None:
                self.render_anomaly_detection_results(st.session_state.datasets['target']['data'], 
                                                    st.session_state.anomaly_results)
            else:
                st.info("No anomaly detection results available. Please run anomaly detection analysis first.")
                
                if st.button("Run Anomaly Detection Analysis"):
                    if st.session_state.datasets is not None:
                        anomaly_results = self.run_anomaly_detection(config, st.session_state.datasets['target']['data'])
                    else:
                        st.warning("Please generate data first.")
        
        with tab5:
            if st.session_state.trainer is not None:
                self.render_training_results(st.session_state.trainer)
            else:
                st.info("No training results available. Please train a model first.")
                
                if st.button("Train Model"):
                    if st.session_state.datasets is not None:
                        trainer = self.train_model(config, st.session_state.datasets)
                    else:
                        st.warning("Please generate data first.")


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
