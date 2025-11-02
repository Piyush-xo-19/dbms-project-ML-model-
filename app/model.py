import joblib
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class BreastCancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_or_train_model()
        
    def load_or_train_model(self):
        """Load existing model or train a new one if models don't exist"""
        try:
            # Try to load existing models
            if os.path.exists("svm_model.pkl") and os.path.exists("scaler.pkl"):
                self.model = joblib.load("svm_model.pkl")
                self.scaler = joblib.load("scaler.pkl")
                print("Loaded existing SVM model and scaler")
            else:
                print("Model files not found. Training new model...")
                self.train_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            self.train_model()
            
    def train_model(self):
        """Train a new SVM model using the breast cancer dataset"""
        try:
            # Load the breast cancer dataset
            data = load_breast_cancer()
            X, y = data.data, data.target
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train SVM model
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate the model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            print(f"Model trained successfully!")
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Testing accuracy: {test_score:.4f}")
            
            # Save the model and scaler
            joblib.dump(self.model, "svm_model.pkl")
            joblib.dump(self.scaler, "scaler.pkl")
            print("Model and scaler saved successfully!")
            
        except Exception as e:
            print(f"Error training model: {e}")
            raise
            
    def predict(self, features):
        """
        Make a prediction on the given features
        
        Args:
            features (list): List of 30 feature values
            
        Returns:
            tuple: (prediction, confidence) where prediction is 0 (benign) or 1 (malignant)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded or trained")
            
        if len(features) != 30:
            raise ValueError(f"Expected 30 features, got {len(features)}")
            
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get prediction probabilities for confidence
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
        
    def get_feature_names(self):
        """Return the names of the 30 features expected by the model"""
        return [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
            'mean_smoothness', 'mean_compactness', 'mean_concavity',
            'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
            'se_radius', 'se_texture', 'se_perimeter', 'se_area',
            'se_smoothness', 'se_compactness', 'se_concavity',
            'se_concave_points', 'se_symmetry', 'se_fractal_dimension',
            'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
            'worst_smoothness', 'worst_compactness', 'worst_concavity',
            'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
        ]