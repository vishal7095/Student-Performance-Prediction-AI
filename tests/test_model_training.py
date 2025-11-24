"""
Tests for model training module
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import ModelTrainer
from data_processing import DataProcessor

class TestModelTraining:
    """Test model training functionality"""
    
    @pytest.fixture
    def model_trainer(self):
        """Create ModelTrainer instance for testing"""
        return ModelTrainer()
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        data_processor = DataProcessor()
        data = data_processor.generate_sample_data(100)
        processed_data = data_processor.clean_data(data)
        engineered_data = data_processor.engineer_features(processed_data)
        X, y, features = data_processor.prepare_training_data(engineered_data)
        return X, y, features
    
    def test_model_initialization(self, model_trainer):
        """Test model initialization"""
        model_configs = model_trainer.initialize_models()
        
        expected_models = ['linear_regression', 'random_forest', 'gradient_boosting']
        for model_name in expected_models:
            assert model_name in model_configs
            assert 'model' in model_configs[model_name]
    
    def test_model_training(self, model_trainer, sample_training_data):
        """Test model training"""
        X, y, features = sample_training_data
        
        results = model_trainer.train_models(X, y)
        
        assert len(results) > 0
        for model_name, result in results.items():
            if result['model'] is not None:
                assert 'metrics' in result
                assert 'r2' in result['metrics']
                assert result['metrics']['r2'] >= -1  # R² can be negative but should be reasonable
    
    def test_best_model_selection(self, model_trainer, sample_training_data):
        """Test best model selection"""
        X, y, features = sample_training_data
        
        results = model_trainer.train_models(X, y)
        best_model_name = model_trainer.select_best_model()
        
        assert best_model_name in results
        assert results[best_model_name]['model'] is not None
        
        # Best model should have highest R²
        best_r2 = results[best_model_name]['metrics']['r2']
        for model_name, result in results.items():
            if (result['model'] is not None and 
                model_name != best_model_name and 
                'r2' in result['metrics']):
                assert best_r2 >= result['metrics']['r2']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])