import sys
import joblib
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton

# Load pre-trained SVM model
model = joblib.load("model/svm_model.pkl")

class SVMApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVM Prediction App")
        self.setGeometry(200, 200, 300, 200)
        
        layout = QVBoxLayout()

        # Input fields
        self.label1 = QLabel("Feature 1:")
        self.input1 = QLineEdit()
        layout.addWidget(self.label1)
        layout.addWidget(self.input1)

        self.label2 = QLabel("Feature 2:")
        self.input2 = QLineEdit()
        layout.addWidget(self.label2)
        layout.addWidget(self.input2)

        # Predict button
        self.button = QPushButton("Predict")
        self.button.clicked.connect(self.predict)
        layout.addWidget(self.button)

        # Output label
        self.result_label = QLabel("Result will appear here")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict(self):
        try:
            f1 = float(self.input1.text())
            f2 = float(self.input2.text())
            prediction = model.predict(np.array([[f1, f2]]))[0]
            self.result_label.setText(f"Prediction: {prediction}")
        except ValueError:
            self.result_label.setText("Please enter valid numbers.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SVMApp()
    window.show()
    sys.exit(app.exec())
