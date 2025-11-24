"""
Tests for data processing module
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor
from utils import Config

class TestDataProcessing:
    """Test data processing functionality"""
    
    @pytest.fixture
    def data_processor(self):
        """Create DataProcessor instance for testing"""
        return DataProcessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'student_id': ['STU001', 'STU002', 'STU003'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'attendance': [85, 92, 78],
            'study_hours': [5.0, 7.5, 3.0],
            'previous_score': [75.0, 88.0, 65.0],
            'assignment_score': [80.0, 90.0, 70.0],
            'participation': [7, 9, 5],
            'final_score': [78.0, 89.0, 68.0]
        })
    
    def test_generate_sample_data(self, data_processor):
        """Test sample data generation"""
        data = data_processor.generate_sample_data(100)
        
        assert len(data) == 100
        assert 'student_id' in data.columns
        assert 'final_score' in data.columns
        assert data['attendance'].between(60, 100).all()
        assert data['study_hours'].between(1, 10).all()
    
    def test_clean_data(self, data_processor, sample_data):
        """Test data cleaning"""
        # Add some problematic data
        sample_data.loc[3] = ['STU004', 'Duplicate', 85, 5.0, 75.0, 80.0, 7, 78.0]  # Duplicate
        sample_data.loc[4] = ['STU005', 'With NaN', 85, np.nan, 75.0, 80.0, 7, 78.0]  # Missing value
        
        cleaned_data = data_processor.clean_data(sample_data)
        
        assert len(cleaned_data) == len(sample_data) - 1  # Duplicate removed
        assert cleaned_data['study_hours'].isna().sum() == 0  # NaN filled
    
    def test_engineer_features(self, data_processor, sample_data):
        """Test feature engineering"""
        engineered_data = data_processor.engineer_features(sample_data)
        
        expected_features = ['attendance_ratio', 'study_efficiency', 'academic_base']
        for feature in expected_features:
            assert feature in engineered_data.columns
        
        assert engineered_data['attendance_ratio'].between(0, 1).all()
    
    def test_validate_data(self, data_processor, sample_data):
        """Test data validation"""
        is_valid, errors = data_processor.validate_data(sample_data)
        
        assert is_valid
        assert len(errors) == 0
        
        # Test with invalid data
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'attendance'] = 150  # Invalid value
        
        is_valid, errors = data_processor.validate_data(invalid_data)
        assert not is_valid
        assert any('attendance' in error for error in errors)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])