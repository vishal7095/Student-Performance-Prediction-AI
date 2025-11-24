"""
Utility functions for the Student Performance Predictor
"""
import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def save(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

def create_directories() -> None:
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'data/feedback',
        'models/trained_models',
        'models/model_history',
        'src',
        'tests',
        'notebooks',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def generate_student_id(name: str, index: int) -> str:
    """Generate unique student ID"""
    hash_input = f"{name}_{index}_{datetime.now().timestamp()}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:8].upper()

def calculate_final_score(row: pd.Series) -> float:
    """
    Calculate final score based on multiple factors with realistic weights
    """
    base_score = (
        row['previous_score'] * 0.4 +  # Previous academic performance
        row['assignment_score'] * 0.25 +  # Current assignment performance
        row['attendance'] * 0.15 +  # Attendance impact
        row['study_hours'] * 2.5 +  # Study hours impact (2.5 points per hour)
        row['participation'] * 1.2   # Participation impact
    )
    
    # Add some random noise to make it realistic
    noise = np.random.normal(0, 3)
    final_score = base_score + noise
    
    # Ensure score is within bounds
    final_score = max(50, min(100, final_score))
    
    return round(final_score, 1)

def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate student data for quality"""
    errors = []
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        errors.append("Dataset contains missing values")
    
    # Check score ranges
    score_columns = ['attendance', 'assignment_score', 'previous_score', 'final_score']
    for col in score_columns:
        if col in df.columns:
            if (df[col] < 0).any() or (df[col] > 100).any():
                errors.append(f"Invalid values in {col}: must be between 0-100")
    
    # Check study hours
    if 'study_hours' in df.columns:
        if (df['study_hours'] < 0).any() or (df['study_hours'] > 24).any():
            errors.append("Invalid study hours: must be between 0-24")
    
    # Check participation
    if 'participation' in df.columns:
        if (df['participation'] < 1).any() or (df['participation'] > 10).any():
            errors.append("Invalid participation: must be between 1-10")
    
    return len(errors) == 0, errors

def save_model_history(model_name: str, metrics: Dict[str, float], 
                      features: List[str], timestamp: str) -> None:
    """Save model training history"""
    history_file = 'models/model_history/model_versions.json'
    
    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            history = json.load(file)
    else:
        history = {}
    
    # Add new entry
    history[timestamp] = {
        'model_name': model_name,
        'metrics': metrics,
        'features': features,
        'timestamp': timestamp
    }
    
    # Save updated history
    with open(history_file, 'w') as file:
        json.dump(history, file, indent=2)
    
    logger.info(f"Model history saved for {model_name}")

def load_model_history() -> Dict[str, Any]:
    """Load model training history"""
    history_file = 'models/model_history/model_versions.json'
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            return json.load(file)
    return {}

class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self):
        self.performance_file = 'models/model_history/performance_tracking.json'
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create performance tracking file if it doesn't exist"""
        if not os.path.exists(self.performance_file):
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            with open(self.performance_file, 'w') as file:
                json.dump({}, file)
    
    def track_performance(self, model_name: str, metrics: Dict[str, float],
                         dataset_size: int, timestamp: str):
        """Track model performance"""
        with open(self.performance_file, 'r') as file:
            data = json.load(file)
        
        if model_name not in data:
            data[model_name] = []
        
        data[model_name].append({
            'timestamp': timestamp,
            'metrics': metrics,
            'dataset_size': dataset_size
        })
        
        with open(self.performance_file, 'w') as file:
            json.dump(data, file, indent=2)
        
        logger.info(f"Performance tracked for {model_name}")

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1%}"

def format_score(score: float) -> str:
    """Format score with 1 decimal place"""
    return f"{score:.1f}"

def get_feature_importance_color(importance: float) -> str:
    """Get color for feature importance visualization"""
    if importance >= 0.3:
        return "#FF6B6B"  # High importance - red
    elif importance >= 0.15:
        return "#4ECDC4"  # Medium importance - teal
    else:
        return "#45B7D1"  # Low importance - blue