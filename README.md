# Student Performance Prediction AI

An intelligent machine learning system that predicts student academic performance and continuously improves through user feedback.

## 🎯 Overview

This AI-powered system analyzes student behavioral and academic data to forecast final exam scores. The system learns from teacher and student feedback to make increasingly accurate predictions over time.

## ✨ Features

- **📊 Smart Predictions**: Uses machine learning to predict student final scores
- **🎓 Multi-factor Analysis**: Considers attendance, study hours, previous scores, assignments, and participation
- **🔄 Continuous Learning**: Improves predictions through user feedback integration
- **🌐 Web Interface**: User-friendly Streamlit application for easy access
- **📈 Visualization**: Comprehensive dashboards and performance analytics
- **⚙️ Configurable**: Flexible settings through YAML configuration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/vishal7095/Student-Performance-Prediction-AI.git
cd Student-Performance-Prediction-AI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Initialize the system**
```bash
python main.py --init
```

4. **Train the model**
```bash
python main.py --train
```

5. **Launch the web application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
Student-Performance-Prediction-AI/
├── data/
│   ├── raw/                 # Raw student data
│   ├── processed/           # Cleaned and processed data
│   └── feedback/            # User feedback data
├── models/
│   ├── trained_models/      # Saved ML models
│   └── model_history/       # Model version history
├── src/                    # Source code
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Test cases
├── config/                 # Configuration files
├── app.py                 # Streamlit web application
└── main.py               # Command-line interface
```

## 🛠️ Usage

### Web Application
Access the system through the Streamlit web interface:
```bash
streamlit run app.py
```

### Command Line Interface
```bash
# Generate sample data and initialize system
python main.py --init

# Train a new model
python main.py --train

# Run demonstration
python main.py --demo

# Check system status
python main.py --status

# Generate reports
python main.py --reports
```

## 🔧 Configuration

Modify `config/config.yaml` to customize:
- Data generation parameters
- Model training settings
- Feedback system weights
- Application settings

## 🤖 Machine Learning

The system employs multiple algorithms and automatically selects the best performer:
- **Random Forest**: Robust handling of non-linear patterns
- **Gradient Boosting**: High accuracy for complex relationships
- **Linear Models**: Fast and interpretable predictions

## 📊 Sample Prediction

Input features required:
- Attendance percentage
- Daily study hours
- Previous exam scores
- Assignment scores
- Class participation level

Output includes:
- Predicted final score
- Confidence level
- Performance category
- Personalized recommendations
- Feature importance analysis

## 🎯 Feedback System

The system improves through:
1. **Rating Collection**: Users rate prediction accuracy (1-5 stars)
2. **Actual Scores**: Optional submission of real exam results
3. **Pattern Analysis**: Identifies common prediction issues
4. **Automatic Retraining**: Incorporates feedback into new models

## 🧪 Testing

Run the test suite to ensure everything works correctly:
```bash
pytest tests/ -v
```

## 📈 Performance Monitoring

The system provides comprehensive monitoring:
- Model performance metrics (R², MAE, RMSE)
- Feature importance analysis
- Feedback statistics and patterns
- System health checks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
