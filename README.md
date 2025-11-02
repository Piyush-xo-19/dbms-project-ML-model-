# Breast Cancer Prediction with SVM

This project integrates a **Support Vector Machine (SVM)** model with a modern **PyQt6 desktop application** for breast cancer diagnosis prediction. It demonstrates machine learning model training, deployment, and a professional user interface for medical data analysis.

---

## 🔹 Project Overview
- Data preprocessing with **StandardScaler**  
- Training a **Support Vector Classifier (SVC)** on breast cancer dataset
- **PyQt6 desktop application** with modern dark theme interface
- Real-time prediction with confidence scoring
- Professional medical software styling with glassmorphism effects

---

## 🔹 Features
- **Machine Learning**: SVM classifier for breast cancer prediction
- **Modern UI**: PyQt6 application with responsive design and dark theme
- **Data Input**: Organized input fields for 30 cellular measurements
- **Sample Data**: Generate realistic test data with reproducible seeds
- **Real-time Analysis**: Instant predictions with confidence percentages
- **Professional Design**: Medical-grade interface with proper alignment and styling

---

## 🔹 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Piyush-xo-19/dbms-project-ML-model-.git
   cd dbms-project-ML-model-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the desktop application**:
   ```bash
   python app/main.py
   ```

4. **Optional - Train custom model**:
   ```bash
   jupyter notebook svm1_.ipynb
   ```

---

## 🔹 Application Usage

### Input Methods
- **Manual Entry**: Enter cellular measurement values in organized input sections
- **Generate Data**: Use seed-based random data generation for testing
- **Sample Data**: Load pre-configured test cases

### Measurement Categories
- **Mean Values**: Radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- **Standard Error**: Variability measurements for all parameters  
- **Worst Values**: Maximum values observed in the sample

### Analysis
- Click **Analyze** to get AI prediction
- Results show **Benign** (green) or **Malignant** (red) with confidence percentage
- Detailed recommendations provided based on prediction

---

## 🔹 Technical Details

### Project Structure
```
app/
├── main.py          # PyQt6 desktop application
├── model.py         # ML model wrapper and prediction logic
├── svm_model.pkl    # Trained SVM classifier
└── scaler.pkl       # Feature scaling transformer

data/
└── data.csv         # Breast cancer dataset

svm1_.ipynb          # Model training notebook
requirements.txt     # Python dependencies
```

### Technologies Used
- **PyQt6**: Modern cross-platform GUI framework
- **scikit-learn**: Machine learning model and preprocessing
- **NumPy**: Numerical computations
- **joblib**: Model serialization

---

## 👨‍💻 Contributors
- [**Piyush Gupta**](https://github.com/Piyush-xo-19)  
- [**Manandeep Singh**](https://github.com/ManandeepSingh1196)  

---