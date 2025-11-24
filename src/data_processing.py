"""
Data processing module for Student Performance Predictor
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from scipy import stats
import warnings

from .utils import Config, calculate_final_score, generate_student_id, validate_data

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle all data processing operations"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.raw_data_path = self.config.get('data.raw_path')
        self.processed_data_path = self.config.get('data.processed_path')
    
    def generate_sample_data(self, num_students: Optional[int] = None) -> pd.DataFrame:
        """Generate realistic sample student data"""
        if num_students is None:
            num_students = self.config.get('data.generation.num_students', 1000)
        
        np.random.seed(self.config.get('data.generation.random_seed', 42))
        
        # Student names pool
        first_names = [
            'John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah', 'Chris', 'Amanda',
            'Robert', 'Jennifer', 'Daniel', 'Lisa', 'Matthew', 'Michelle', 'Kevin', 'Laura',
            'James', 'Amy', 'Thomas', 'Kimberly', 'Richard', 'Jessica', 'Charles', 'Angela',
            'Christopher', 'Melissa', 'Steven', 'Rebecca', 'Brian', 'Stephanie',
            'Aarav', 'Priya', 'Wei', 'Mei', 'Hiroshi', 'Yuki', 'Raj', 'Ananya',
            'Mohammed', 'Fatima', 'Juan', 'Maria', 'Alexei', 'Svetlana', 'Kwame', 'Ama'
        ]
        
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
            'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
            'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
            'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson',
            'Walker', 'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen',
            'Hill', 'Flores', 'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera'
        ]
        
        data = []
        
        for i in range(num_students):
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            name = f"{first_name} {last_name}"
            student_id = generate_student_id(name, i)
            
            # Generate features with realistic correlations
            attendance = np.random.normal(85, 10)
            attendance = max(60, min(100, attendance))
            
            study_hours = np.random.normal(5, 2)
            study_hours = max(1, min(10, round(study_hours, 1)))
            
            previous_score = np.random.normal(75, 12)
            previous_score = max(50, min(95, round(previous_score, 1)))
            
            assignment_score = previous_score + np.random.normal(0, 8)
            assignment_score = max(50, min(100, round(assignment_score, 1)))
            
            participation = np.random.normal(6, 2)
            participation = max(1, min(10, int(participation)))
            
            # Create row and calculate final score
            row = {
                'student_id': student_id,
                'name': name,
                'attendance': attendance,
                'study_hours': study_hours,
                'previous_score': previous_score,
                'assignment_score': assignment_score,
                'participation': participation
            }
            
            # Calculate final score
            row['final_score'] = calculate_final_score(row)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Validate generated data
        is_valid, errors = validate_data(df)
        if not is_valid:
            logger.warning(f"Data validation errors: {errors}")
        
        logger.info(f"Generated sample data for {num_students} students")
        return df
    
    def save_raw_data(self, df: pd.DataFrame) -> None:
        """Save raw data to CSV"""
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        df.to_csv(self.raw_data_path, index=False)
        logger.info(f"Raw data saved to {self.raw_data_path}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        if not os.path.exists(self.raw_data_path):
            logger.warning("Raw data file not found. Generating sample data...")
            df = self.generate_sample_data()
            self.save_raw_data(df)
            return df
        
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"Raw data loaded from {self.raw_data_path}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        # Create a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Handle missing values
        numeric_columns = ['attendance', 'study_hours', 'previous_score', 
                          'assignment_score', 'participation', 'final_score']
        
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Remove duplicates
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['student_id'], keep='first')
        duplicates_removed = initial_count - len(cleaned_df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate student records")
        
        # Handle outliers using IQR method
        for col in numeric_columns:
            if col in cleaned_df.columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                cleaned_df[col] = np.where(cleaned_df[col] < lower_bound, lower_bound, cleaned_df[col])
                cleaned_df[col] = np.where(cleaned_df[col] > upper_bound, upper_bound, cleaned_df[col])
        
        # Ensure data types are correct
        cleaned_df['student_id'] = cleaned_df['student_id'].astype(str)
        if 'name' in cleaned_df.columns:
            cleaned_df['name'] = cleaned_df['name'].astype(str)
        
        # Validate cleaned data
        is_valid, errors = validate_data(cleaned_df)
        if not is_valid:
            logger.error(f"Data validation failed after cleaning: {errors}")
            raise ValueError(f"Data validation errors: {errors}")
        
        logger.info(f"Data cleaning completed. Final dataset: {len(cleaned_df)} students")
        return cleaned_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better prediction"""
        engineered_df = df.copy()
        
        # Academic performance indicators
        engineered_df['attendance_ratio'] = engineered_df['attendance'] / 100
        engineered_df['study_efficiency'] = engineered_df['study_hours'] * engineered_df['participation']
        
        # Performance trends
        engineered_df['improvement_potential'] = (
            (100 - engineered_df['previous_score']) * engineered_df['attendance_ratio']
        )
        
        # Combined academic score
        engineered_df['academic_base'] = (
            engineered_df['previous_score'] * 0.6 + 
            engineered_df['assignment_score'] * 0.4
        )
        
        # Risk indicator (students who might need help)
        engineered_df['at_risk'] = (
            (engineered_df['attendance'] < 70) | 
            (engineered_df['previous_score'] < 60) |
            (engineered_df['study_hours'] < 3)
        ).astype(int)
        
        # Performance category based on previous scores
        def get_performance_category(score):
            if score >= 85:
                return 'Excellent'
            elif score >= 70:
                return 'Good'
            elif score >= 60:
                return 'Average'
            else:
                return 'Needs Improvement'
        
        engineered_df['performance_category'] = engineered_df['previous_score'].apply(get_performance_category)
        
        logger.info("Feature engineering completed")
        return engineered_df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features and target for model training"""
        # Select features for training
        base_features = [
            'attendance', 'study_hours', 'previous_score', 
            'assignment_score', 'participation'
        ]
        
        engineered_features = [
            'attendance_ratio', 'study_efficiency', 
            'improvement_potential', 'academic_base'
        ]
        
        # Use available features
        available_features = [f for f in base_features + engineered_features if f in df.columns]
        
        X = df[available_features]
        y = df['final_score']
        
        logger.info(f"Prepared training data with {len(available_features)} features")
        logger.info(f"Features: {available_features}")
        
        return X, y, available_features
    
    def save_processed_data(self, df: pd.DataFrame) -> None:
        """Save processed data to CSV"""
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        df.to_csv(self.processed_data_path, index=False)
        logger.info(f"Processed data saved to {self.processed_data_path}")
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load processed data from CSV"""
        if not os.path.exists(self.processed_data_path):
            logger.warning("Processed data not found. Processing raw data...")
            raw_df = self.load_raw_data()
            cleaned_df = self.clean_data(raw_df)
            engineered_df = self.engineer_features(cleaned_df)
            self.save_processed_data(engineered_df)
            return engineered_df
        
        df = pd.read_csv(self.processed_data_path)
        logger.info(f"Processed data loaded from {self.processed_data_path}")
        return df

    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset"""
        stats = {}
        
        # Basic info
        stats['total_students'] = len(df)
        stats['total_features'] = len(df.columns)
        stats['feature_names'] = list(df.columns)
        
        # Numeric columns statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing': int(df[col].isnull().sum())
            }
        
        # Correlation with final score
        if 'final_score' in numeric_columns:
            correlations = {}
            for col in numeric_columns:
                if col != 'final_score':
                    corr = df[col].corr(df['final_score'])
                    correlations[col] = float(corr)
            
            stats['correlations_with_final_score'] = correlations
        
        # Performance categories if available
        if 'performance_category' in df.columns:
            category_counts = df['performance_category'].value_counts().to_dict()
            stats['performance_categories'] = category_counts
        
        # Risk analysis if available
        if 'at_risk' in df.columns:
            at_risk_count = int(df['at_risk'].sum())
            stats['at_risk_students'] = at_risk_count
            stats['at_risk_percentage'] = float(at_risk_count / len(df))
        
        logger.info("Generated comprehensive data statistics")
        return stats