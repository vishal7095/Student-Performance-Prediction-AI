"""
Tests for prediction module
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import Predictor
from model_training import ModelTrainer
from data_processing import DataProcessor

class TestPrediction:
    """Test prediction functionality"""
    
    @pytest.fixture
    def predictor(self):
        """Create Predictor instance for testing"""
        return Predictor()
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input data"""
        return {
            'attendance': 85,
            'study_hours': 5.0,
            'previous_score': 75.0,
            'assignment_score': 80.0,
            'participation': 7
        }
    
    def test_input_validation(self, predictor, sample_input):
        """Test input validation"""
        is_valid, errors = predictor.validate_prediction_input(sample_input)
        
        assert is_valid
        assert len(errors) == 0
        
        # Test with missing feature
        invalid_input = sample_input.copy()
        del invalid_input['attendance']
        
        is_valid, errors = predictor.validate_prediction_input(invalid_input)
        assert not is_valid
        assert any('attendance' in error for error in errors)
        
        # Test with invalid value
        invalid_input = sample_input.copy()
        invalid_input['attendance'] = 150  # Out of range
        
        is_valid, errors = predictor.validate_prediction_input(invalid_input)
        assert not is_valid
        assert any('attendance' in error for error in errors)
    
    def test_prediction_output(self, predictor, sample_input):
        """Test prediction output structure"""
        # This test requires a trained model
        if predictor.model is None:
            pytest.skip("No trained model available")
        
        result = predictor.predict_single(sample_input)
        
        expected_keys = ['predicted_score', 'confidence', 'performance_category', 
                        'feature_contributions', 'recommendations']
        
        for key in expected_keys:
            assert key in result
        
        assert 0 <= result['predicted_score'] <= 100
        assert 0 <= result['confidence'] <= 1
        assert isinstance(result['recommendations'], list)
    
    def test_batch_prediction(self, predictor):
        """Test batch prediction"""
        if predictor.model is None:
            pytest.skip("No trained model available")
        
        batch_input = [
            {
                'attendance': 85,
                'study_hours': 5.0,
                'previous_score': 75.0,
                'assignment_score': 80.0,
                'participation': 7
            },
            {
                'attendance': 92,
                'study_hours': 7.5,
                'previous_score': 88.0,
                'assignment_score': 90.0,
                'participation': 9
            }
        ]
        
        results = predictor.predict_batch(batch_input)
        
        assert len(results) == len(batch_input)
        for result in results:
            if 'error' not in result:
                assert 'predicted_score' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])