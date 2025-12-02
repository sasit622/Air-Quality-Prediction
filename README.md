# 🌍 Air Quality Prediction App

A comprehensive machine learning application that predicts Air Quality Index (AQI) values and categorizes air quality based on pollutant concentrations. Built with multiple ML models including Linear Regression, Lasso Regression, SVM, and Bayesian Logistic Regression for accurate environmental monitoring.

![Air Quality Dashboard](images/dashboard.png)

## 🌟 Overview

Air pollution is a critical environmental issue affecting public health worldwide. This application uses machine learning to predict AQI values based on concentrations of major air pollutants, helping users understand and monitor air quality conditions.

The system analyzes six key pollutants:
- **PM2.5** - Fine Particulate Matter (≤2.5 micrometers)
- **PM10** - Coarse Particulate Matter (≤10 micrometers)
- **NO2** - Nitrogen Dioxide
- **SO2** - Sulfur Dioxide
- **CO** - Carbon Monoxide
- **Ozone** - Ground-level Ozone

## ✨ Key Features

### 🎯 Dual Prediction Modes
1. **Exact AQI Value Prediction**
   - Linear Regression model
   - Precise numerical AQI output
   - Best for quantitative analysis

2. **Air Quality Category Classification**
   - Multi-class classification
   - Categories: Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous
   - Best for quick quality assessment

### 📊 Multiple ML Models
- **Linear Regression** - Fast and interpretable predictions
- **Lasso Regression** - Feature selection with regularization
- **Support Vector Machine (SVM)** - High-dimensional classification
- **Bayesian Logistic Regression** - Probabilistic classification

### 🎨 Interactive Web Interface
- **Date & Year Selection**: Optional temporal context
- **Holiday Toggle**: Account for holiday effects on air quality
- **Pollutant Level Controls**: Adjustable input sliders for each pollutant
- **Real-time Predictions**: Instant AQI calculation
- **Prediction History**: Save predictions to CSV file
- **Dark Theme**: Modern, eye-friendly interface

### 📈 Model Analytics
- **Feature Coefficient Visualization**: Understand pollutant impact
- **Model Performance Comparison**: Accuracy, Precision, Recall, F1-Score
- **Actual vs Predicted Charts**: Lasso regression performance visualization
- **Feature Importance Analysis**: Identify key contributing factors

## 🛠️ Technology Stack

### Machine Learning
- **scikit-learn** - ML models and preprocessing
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

### Web Framework
- **Streamlit** - Interactive web application
- **Python 3.8+** - Core programming language

### Data Processing
- **pickle** - Model serialization
- **csv** - Prediction history storage

## 📸 Screenshots

### Main Dashboard
![Dashboard Interface](images/dashboard-interface.png)

### Feature Coefficients
![Feature Coefficients](images/feature-coefficients.png)
*Shows the relative importance of each pollutant in AQI prediction*

### Model Performance
![Model Comparison](images/model-performance.png)
*Comparison of different ML models across multiple metrics*

### Lasso Regression Analysis
![Actual vs Predicted](images/lasso-regression.png)
*Scatter plot showing prediction accuracy*

## 🚀 Installation & Setup

### **Prerequisites**
```bash
# Python 3.8 or higher
python --version

# pip package manager
pip --version
```

### **Installation Steps**
```bash
# 1. Clone the repository
git clone https://github.com/sasit622/Air-Quality-Prediction.git
cd Air-Quality-Prediction

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📁 Project Structure
```
Air-Quality-Prediction/
├── app.py                          # Main Streamlit application
├── models/
│   ├── linear_regression.pkl      # Trained Linear Regression model
│   ├── lasso_regression.pkl       # Trained Lasso Regression model
│   ├── svm_model.pkl              # Trained SVM model
│   ├── bayesian_model.pkl         # Trained Bayesian Logistic model
│   └── scaler.pkl                 # Feature scaler
├── data/
│   ├── city_day.csv               # Training dataset
│   └── processed_data.csv         # Cleaned and preprocessed data
├── notebooks/
│   ├── data_exploration.ipynb     # EDA notebook
│   ├── model_training.ipynb       # Model training experiments
│   └── model_evaluation.ipynb     # Performance analysis
├── utils/
│   ├── preprocessing.py           # Data preprocessing functions
│   ├── model_utils.py             # Model loading and prediction
│   └── visualization.py           # Plotting functions
├── images/                        # Screenshots and visualizations
├── prediction_history.csv         # Saved predictions
├── requirements.txt               # Python dependencies
└── README.md
```

## 🎯 How It Works

### **1. Data Collection**
The model is trained on historical air quality data containing:
- Pollutant concentrations (PM2.5, PM10, NO2, SO2, CO, Ozone)
- Temporal features (Date, Year, Holiday indicator)
- Target variable (AQI value or category)

### **2. Data Preprocessing**
```python
# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

### **3. Model Training**

#### Linear Regression
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

#### Lasso Regression
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

#### SVM Classification
```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
```

#### Bayesian Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
bayesian = LogisticRegression(penalty='l2', solver='saga')
bayesian.fit(X_train, y_train)
```

### **4. Prediction Pipeline**
```python
# User input
input_data = {
    'PM2.5': 50.0,
    'PM10': 50.0,
    'NO2': 20.0,
    'SO2': 10.0,
    'CO': 1.0,
    'Ozone': 0.05
}

# Scale input
scaled_input = scaler.transform([list(input_data.values())])

# Predict
aqi_prediction = model.predict(scaled_input)[0]
```

### **5. AQI Categories**

| AQI Range | Category | Color | Health Implications |
|-----------|----------|-------|---------------------|
| 0-50 | Good | 🟢 Green | Air quality is satisfactory |
| 51-100 | Moderate | 🟡 Yellow | Acceptable for most people |
| 101-150 | Unhealthy for Sensitive Groups | 🟠 Orange | May affect sensitive individuals |
| 151-200 | Unhealthy | 🔴 Red | Everyone may experience health effects |
| 201-300 | Very Unhealthy | 🟣 Purple | Health alert for everyone |
| 301+ | Hazardous | 🟤 Maroon | Health emergency |

## 📊 Model Performance

### **Regression Models (AQI Value Prediction)**

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 28.45 | 42.67 | 0.89 |
| Lasso Regression | 29.12 | 43.21 | 0.88 |

### **Classification Models (Category Prediction)**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM | 0.97 | 0.96 | 0.86 | 0.86 |
| Bayesian Logistic | 0.94 | 0.93 | 0.85 | 0.86 |

*Metrics based on test set evaluation*

### **Feature Importance**

Based on coefficient analysis:
1. **CO (Carbon Monoxide)** - Highest impact (coefficient: ~13.9)
2. **PM10** - Moderate impact (coefficient: ~0.6)
3. **PM2.5** - Moderate impact (coefficient: ~0.4)
4. **SO2** - Low impact (coefficient: ~0.1)
5. **NO2** - Low impact (coefficient: ~0.2)
6. **Ozone** - Minimal impact (coefficient: ~0.05)

![Feature Coefficients](images/feature-coefficients.png)

## 🎨 User Interface Features

### **Input Controls**
```python
# Date selector
date = st.date_input("Date (Optional)")

# Year input with increment/decrement
year = st.number_input("Year", min_value=2000, max_value=2030)

# Holiday checkbox
is_holiday = st.checkbox("Is it a Holiday?")

# Pollutant sliders
pm25 = st.slider("PM2.5", min_value=0.0, max_value=500.0, step=0.1)
pm10 = st.slider("PM10", min_value=0.0, max_value=500.0, step=0.1)
no2 = st.slider("NO2", min_value=0.0, max_value=200.0, step=0.1)
so2 = st.slider("SO2", min_value=0.0, max_value=100.0, step=0.1)
co = st.slider("CO", min_value=0.0, max_value=50.0, step=0.1)
ozone = st.slider("Ozone", min_value=0.0, max_value=0.5, step=0.01)
```

### **Prediction Display**
- **Exact AQI Value**: Shows numerical prediction (e.g., "93.62")
- **Air Quality Category**: Shows classification (e.g., "Moderate")
- **Color-coded Results**: Visual feedback based on category
- **Save Button**: Export predictions with timestamp

## 💾 Prediction History

Predictions are saved to `prediction_history.csv` with:
- Timestamp
- Date and Year (if provided)
- All pollutant values
- Predicted AQI or Category
- Model used
```csv
Timestamp,Date,Year,PM2.5,PM10,NO2,SO2,CO,Ozone,Prediction,Model
2025-12-02 10:30:45,2025-12-02,2025,50.0,50.0,20.0,10.0,1.0,0.05,93.62,Linear Regression
```

## 🔧 Configuration

### **Model Parameters**

Edit `config.py` to adjust:
```python
# Linear Regression
LINEAR_REG_PARAMS = {
    'fit_intercept': True,
    'normalize': False
}

# Lasso Regression
LASSO_PARAMS = {
    'alpha': 0.1,
    'max_iter': 1000
}

# SVM
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale'
}
```

### **Pollutant Ranges**

Adjust input ranges in `app.py`:
```python
POLLUTANT_RANGES = {
    'PM2.5': (0, 500),
    'PM10': (0, 500),
    'NO2': (0, 200),
    'SO2': (0, 100),
    'CO': (0, 50),
    'Ozone': (0, 0.5)
}
```

## 🌐 Use Cases

- **Public Health Monitoring**: Track air quality trends
- **Urban Planning**: Inform policy decisions
- **Personal Health**: Plan outdoor activities
- **Research**: Environmental studies and analysis
- **Education**: Learn about air pollution and ML
- **Smart Cities**: Integration with IoT sensors

## 🔮 Future Enhancements

- [ ] Real-time data integration with API
- [ ] Geospatial mapping of AQI predictions
- [ ] Mobile application (iOS/Android)
- [ ] Deep learning models (LSTM, GRU for time series)
- [ ] Historical trend analysis and forecasting
- [ ] Multi-city comparison dashboard
- [ ] Alert system for unhealthy AQI levels
- [ ] Export visualizations as PDF reports
- [ ] Integration with weather data
- [ ] Social media sharing of predictions
- [ ] API endpoint for external applications
- [ ] Docker containerization

## 📚 Dataset Information

### **Source**
Data collected from various air quality monitoring stations across multiple cities.

### **Features**
- **Temporal**: Date, Year, Holiday
- **Pollutants**: PM2.5, PM10, NO2, SO2, CO, Ozone (µg/m³)
- **Target**: AQI value (0-500) or Category

### **Data Statistics**
- Total samples: ~10,000 records
- Time period: 2015-2024
- Cities covered: 50+
- Missing values: <5% (imputed with mean)

## 🧪 Model Training

### **Training Process**
```bash
# Train all models
python train_models.py

# Train specific model
python train_models.py --model linear

# Hyperparameter tuning
python tune_hyperparameters.py
```

### **Evaluation**
```bash
# Evaluate all models
python evaluate_models.py

# Generate performance plots
python visualize_results.py
```

## 🐛 Troubleshooting

### **Common Issues**

**ModuleNotFoundError:**
```bash
pip install -r requirements.txt
```

**Streamlit not opening:**
```bash
streamlit run app.py --server.port 8502
```

**Model file not found:**
```bash
# Retrain models
python train_models.py
```

**Prediction errors:**
- Ensure all pollutant values are within valid ranges
- Check that models are properly loaded
- Verify input data types

## 📖 References

- [EPA AQI Guidelines](https://www.airnow.gov/aqi/)
- [WHO Air Quality Guidelines](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)
- scikit-learn Documentation
- Streamlit Documentation

## 👤 Author

**Sasitharan S**
- GitHub: [@sasit622](https://github.com/sasit622)
- Focus: Machine Learning & Environmental Data Science

## 🙏 Acknowledgments

- Air quality monitoring agencies for data
- scikit-learn team for ML tools
- Streamlit team for the web framework
- Environmental science community
- Open-source contributors

prediction, machine-learning, scikit-learn, streamlit, environmental-science, pollution-monitoring, data-science, python, svm, linear-regression
