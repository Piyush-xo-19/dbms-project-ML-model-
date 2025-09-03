# DBMS Project with Machine Learning (SVM Model)

This project integrates a **Support Vector Machine (SVM)** model into a DBMS-based workflow.  
It demonstrates how to preprocess data, train a machine learning model, and prepare it for deployment as part of a larger application.

---

## 🔹 Project Overview
- Data preprocessing with **StandardScaler**  
- Training a **Support Vector Classifier (SVC)**  
- Exporting both the trained model and scaler using `joblib`  
- Ready to be integrated into a desktop application with **PyQt**  

---

## 🔹 Features
- Loads and preprocesses medical dataset (from `data.csv`)  
- Encodes categorical labels (`diagnosis`)  
- Standardizes features for optimal SVM performance  
- Trains an **SVM classifier** to predict diagnoses  
- Exports trained model and scaler for deployment  

---

## 🔹 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Piyush-xo-19/dbms-project-ML-model-.git
   cd dbms-project-ML-model-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook svm1_.ipynb
   ```

4. After training, export the model and scaler:
   ```python
   import joblib
   joblib.dump(svm, "svm_model.pkl")
   joblib.dump(sd, "scaler.pkl")
   ```

5. Use `svm_model.pkl` and `scaler.pkl` in your application (e.g., a PyQt desktop app).

--- 

---

## 👨‍💻 Contributors
- [**Piyush Gupta**](https://github.com/Piyush-xo-19)  
- [**Manandeep Singh**](https://github.com/ManandeepSingh1196)  

---