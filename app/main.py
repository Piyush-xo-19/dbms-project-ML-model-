import sys
import os
import numpy as np
import random
from PyQt6.QtWidgets import (QWidget, QApplication, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QTextEdit, 
                             QGroupBox, QGridLayout, QMessageBox, QFrame,
                             QScrollArea, QSizePolicy, QSpacerItem)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QRect
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QPainter, QBrush
from model import BreastCancerPredictor

class AnimatedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(50)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.predictor = BreastCancerPredictor()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("SVM Classifier for Breast Cancer Prediction")
        
        # Set larger window size to accommodate non-scrollable text
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1300, 900)
        self.setStyleSheet(self.get_modern_stylesheet())
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self.create_modern_header()
        main_layout.addWidget(header)
        
        # Content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 15)
        content_layout.setSpacing(20)
        
        # Input panel
        input_scroll = self.create_scrollable_input_panel()
        content_layout.addWidget(input_scroll, 3)
        
        # Results panel - give more space for text
        results_panel = self.create_modern_results_panel()
        content_layout.addWidget(results_panel, 2)
        
        main_layout.addWidget(content_widget, 1)
        
        # Footer
        footer = self.create_modern_footer()
        main_layout.addWidget(footer)
        
        self.setLayout(main_layout)
        
    def create_modern_header(self):
        header = QFrame()
        header.setFixedHeight(90)
        
        header.setStyleSheet("""
            QFrame {
                background: #1a1a1a;
                border: none;
                border-bottom: 1px solid #333333;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(3)
        
        title = QLabel("SVM Classifier for Breast Cancer Prediction")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Normal))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #ffffff; background: transparent;")
        
        subtitle = QLabel("AI Medical Analysis")
        subtitle.setFont(QFont("Segoe UI", 10, QFont.Weight.Light))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888888; background: transparent;")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        header.setLayout(layout)
        
        return header
        
    def create_scrollable_input_panel(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #1a1a1a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #444444;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555555;
            }
        """)
        
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(15)
        
        # Create feature groups
        feature_groups = [
            ("Mean Measurements", [
                ("Radius", "mean_radius"), ("Texture", "mean_texture"), 
                ("Perimeter", "mean_perimeter"), ("Area", "mean_area"),
                ("Smoothness", "mean_smoothness"), ("Compactness", "mean_compactness"),
                ("Concavity", "mean_concavity"), ("Concave Points", "mean_concave_points"),
                ("Symmetry", "mean_symmetry"), ("Fractal Dimension", "mean_fractal_dimension")
            ]),
            ("Standard Error", [
                ("Radius SE", "se_radius"), ("Texture SE", "se_texture"),
                ("Perimeter SE", "se_perimeter"), ("Area SE", "se_area"),
                ("Smoothness SE", "se_smoothness"), ("Compactness SE", "se_compactness"),
                ("Concavity SE", "se_concavity"), ("Concave Points SE", "se_concave_points"),
                ("Symmetry SE", "se_symmetry"), ("Fractal Dimension SE", "se_fractal_dimension")
            ]),
            ("Worst Case Values", [
                ("Radius Worst", "worst_radius"), ("Texture Worst", "worst_texture"),
                ("Perimeter Worst", "worst_perimeter"), ("Area Worst", "worst_area"),
                ("Smoothness Worst", "worst_smoothness"), ("Compactness Worst", "worst_compactness"),
                ("Concavity Worst", "worst_concavity"), ("Concave Points Worst", "worst_concave_points"),
                ("Symmetry Worst", "worst_symmetry"), ("Fractal Dimension Worst", "worst_fractal_dimension")
            ])
        ]
        
        self.feature_inputs = {}
        
        for group_title, features in feature_groups:
            group = self.create_feature_group(group_title, features)
            main_layout.addWidget(group)
            
        scroll_area.setWidget(content_widget)
        return scroll_area
        
    def create_feature_group(self, title, features):
        group = QFrame()
        group.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #181818);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }
        """)
        
        layout = QVBoxLayout(group)
        
        # Responsive padding
        padding = max(12, int(self.width() * 0.012))
        layout.setContentsMargins(padding, padding, padding, padding + 5)
        layout.setSpacing(padding)
        
        # Simple group title
        title_clean = title.split(' ', 1)[1] if ' ' in title else title
        title_label = QLabel(title_clean)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        title_label.setStyleSheet("color: #ffffff; background: transparent; border: none; margin-bottom: 8px;")
        layout.addWidget(title_label)
        
        # Features grid with responsive columns
        grid_layout = QGridLayout()
        
        # Responsive spacing and columns
        spacing = max(6, int(self.width() * 0.006))
        grid_layout.setSpacing(spacing)
        
        # Determine columns based on window width
        cols = 3 if self.width() > 1400 else 2
        
        for i, (display_name, field_name) in enumerate(features):
            row = i // cols
            col = i % cols
            
            # Glassmorphism container
            container = QFrame()
            container.setStyleSheet("""
                QFrame {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    backdrop-filter: blur(10px);
                }
                QFrame:hover {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
            """)
            
            container_layout = QVBoxLayout(container)
            inner_padding = max(6, int(self.width() * 0.005))
            container_layout.setContentsMargins(inner_padding, inner_padding, inner_padding, inner_padding)
            container_layout.setSpacing(4)
            
            # Responsive label
            label = QLabel(display_name)
            label_size = max(8, min(11, int(self.width() * 0.007)))
            label.setFont(QFont("Segoe UI", label_size, QFont.Weight.Normal))
            label.setStyleSheet("""
                color: rgba(255, 255, 255, 0.8); 
                background: transparent; 
                border: none;
                letter-spacing: 0.5px;
            """)
            
            # Premium input field
            input_field = QLineEdit()
            input_field.setPlaceholderText("0.00")
            
            input_size = max(9, min(12, int(self.width() * 0.008)))
            input_field.setFont(QFont("Segoe UI", input_size, QFont.Weight.Normal))
            
            # Responsive input height
            input_height = max(28, int(self.height() * 0.035))
            
            input_field.setStyleSheet(f"""
                QLineEdit {{
                    padding: 8px 12px;
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 6px;
                    background: rgba(0, 0, 0, 0.3);
                    color: #ffffff;
                    font-weight: 400;
                    min-height: {input_height}px;
                    selection-background-color: rgba(255, 255, 255, 0.2);
                }}
                QLineEdit:focus {{
                    border: 2px solid rgba(255, 255, 255, 0.4);
                    background: rgba(0, 0, 0, 0.5);
                    box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
                }}
                QLineEdit:hover {{
                    border: 1px solid rgba(255, 255, 255, 0.25);
                    background: rgba(0, 0, 0, 0.4);
                }}
            """)
            
            self.feature_inputs[field_name] = input_field
            
            container_layout.addWidget(label)
            container_layout.addWidget(input_field)
            
            grid_layout.addWidget(container, row, col)
            
        layout.addLayout(grid_layout)
        return group
        
    def create_modern_results_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #181818);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }
        """)
        
        layout = QVBoxLayout(panel)
        
        # Responsive padding
        padding = max(15, int(self.width() * 0.015))
        layout.setContentsMargins(padding, padding, padding, padding)
        layout.setSpacing(padding)
        
        # Simple title
        title = QLabel("Results")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Medium))
        title.setStyleSheet("color: #ffffff; background: transparent; border: none;")
        layout.addWidget(title)
        
        # Result card with premium styling
        result_card = QFrame()
        
        # Responsive result card height
        card_height = max(80, int(self.height() * 0.12))
        result_card.setMinimumHeight(card_height)
        
        result_card.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
            }
        """)
        
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(padding, padding//2, padding, padding//2)
        result_layout.setSpacing(6)
        
        # Responsive result text
        result_size = max(16, min(24, int(self.width() * 0.016)))
        confidence_size = max(10, min(14, int(self.width() * 0.009)))
        
        self.result_label = QLabel("Ready")
        self.result_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Light))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("color: #aaaaaa; background: transparent; border: none;")
        
        self.confidence_label = QLabel("Awaiting input")
        self.confidence_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Light))
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("color: #777777; background: transparent; border: none;")
        
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.confidence_label)
        
        layout.addWidget(result_card)
        
        # Info section with glassmorphism - non-scrollable
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        
        # Remove scroll bars to make it non-scrollable
        self.info_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.info_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Set word wrap and adjust height to fit content
        self.info_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        
        info_size = max(9, min(12, int(self.width() * 0.008)))
        self.info_text.setFont(QFont("Segoe UI", info_size, QFont.Weight.Light))
        
        self.info_text.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
                padding: 12px;
                color: rgba(255, 255, 255, 0.8);
                selection-background-color: rgba(255, 255, 255, 0.2);
            }
        """)
        self.info_text.setPlainText("Enter patient measurements and click Analyze for AI-powered diagnosis.\n\nUse Generate for random test data or Sample for known case.")
        
        layout.addWidget(self.info_text)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def create_modern_footer(self):
        footer = QFrame()
        footer.setFixedHeight(80)
        
        footer.setStyleSheet("""
            QFrame {
                background: #1a1a1a;
                border-top: 1px solid #333333;
            }
        """)
        
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)
        
        # Seed container with glassmorphism
        seed_container = QFrame()
        seed_container.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 4px;
            }
        """)
        
        seed_layout = QHBoxLayout(seed_container)
        seed_layout.setContentsMargins(8, 4, 8, 4)
        seed_layout.setSpacing(12)
        seed_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        # Responsive seed controls
        font_size = max(9, min(12, int(self.width() * 0.008)))
        
        seed_label = QLabel("Seed")
        seed_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
        seed_label.setStyleSheet("color: #aaaaaa; background: transparent;")
        
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("12345")
        
        # Responsive seed input width
        seed_width = max(60, int(self.width() * 0.06))
        self.seed_input.setMaximumWidth(seed_width)
        self.seed_input.setFont(QFont("Segoe UI", 9))
        self.seed_input.setStyleSheet("""
            QLineEdit {
                padding: 6px 8px;
                border: 1px solid #333333;
                border-radius: 3px;
                background: #222222;
                color: #ffffff;
                min-height: 24px;
            }
            QLineEdit:focus {
                border: 1px solid #555555;
            }
        """)
        
        # Premium buttons with consistent styling
        self.generate_btn = AnimatedButton("Generate")
        self.generate_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background: #333333;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-height:16px;
            }
            QPushButton:hover {
                background: #444444;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_random_data)
        
        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.generate_btn)
        
        # Simple action buttons
        self.sample_btn = AnimatedButton("Sample")
        self.sample_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
        self.sample_btn.setStyleSheet("""
            QPushButton {
                background: #333333;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-height: 32px;
            }
            QPushButton:hover {
                background: #444444;
            }
        """)
        self.sample_btn.clicked.connect(self.load_sample_data)
        
        self.clear_btn = AnimatedButton("Clear")
        self.clear_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Normal))
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: #333333;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-height: 32px;
            }
            QPushButton:hover {
                background: #444444;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_inputs)
        
        # Primary analyze button
        self.predict_btn = AnimatedButton("Analyze")
        self.predict_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #000000;
                border: none;
                border-radius: 4px;
                padding: 10px 10px;
                font-weight: 600;
                min-width: 72px;
                min-height: 28px;
            }
            QPushButton:hover {
                background: #f0f0f0;
            }
            QPushButton:pressed {
                background: #e0e0e0;
            }
        """)
        self.predict_btn.clicked.connect(self.predict_diagnosis)
        
        layout.addWidget(seed_container)
        layout.addStretch()
        
        # Button container for better alignment
        button_container = QFrame()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        button_layout.addWidget(self.sample_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.predict_btn)
        
        layout.addWidget(button_container)
        
        return footer
        
    def get_modern_stylesheet(self):
        return """
            QMainWindow, QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0a0a0a, stop:1 #1a1a1a);
                font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
                color: #ffffff;
            }
            
            QMessageBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #181818);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            QMessageBox QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 13px;
                font-weight: 400;
            }
            
            QMessageBox QPushButton {
                background: rgba(255, 255, 255, 0.1);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 500;
                min-width: 80px;
                letter-spacing: 1px;
            }
            
            QMessageBox QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            
            QScrollArea QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.05);
                width: 6px;
                border-radius: 3px;
                margin: 0;
            }
            
            QScrollArea QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                min-height: 20px;
            }
            
            QScrollArea QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            
            QScrollArea QScrollBar::add-line:vertical,
            QScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """
        
    def predict_diagnosis(self):
        try:
            # Collect input values
            features = []
            missing_fields = []
            
            for field_name, input_field in self.feature_inputs.items():
                value = input_field.text().strip()
                if not value:
                    missing_fields.append(field_name)
                    continue
                try:
                    features.append(float(value))
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", f"Invalid value for {field_name}: {value}")
                    return
                    
            if missing_fields:
                QMessageBox.warning(self, "Missing Data", f"Please fill in all fields. Missing: {', '.join(missing_fields[:5])}{'...' if len(missing_fields) > 5 else ''}")
                return
                
            # Make prediction
            prediction, confidence = self.predictor.predict(features)
            
            # Update UI with clean styling
            if prediction == 1:
                self.result_label.setText("Malignant")
                self.result_label.setStyleSheet("""
                    color: #ff6b6b;
                    background: rgba(255, 107, 107, 0.1);
                    border: 1px solid rgba(255, 107, 107, 0.3);
                    border-radius: 6px;
                    font-weight: 400;
                """)
                self.result_label.parent().setStyleSheet("""
                    QFrame {
                        background: rgba(255, 107, 107, 0.05);
                        border-radius: 10px;
                        border: 1px solid rgba(255, 107, 107, 0.2);
                    }
                """)
                info_text = "CRITICAL: AI model predicts malignant case.\n\nImmediate medical consultation required.\nSchedule comprehensive diagnostic tests.\nSeek second medical opinion.\nEarly detection is crucial.\n\nPrediction based on machine learning analysis of cellular characteristics."
            else:
                self.result_label.setText("Benign")
                self.result_label.setStyleSheet("""
                    color: #51cf66;
                    background: rgba(81, 207, 102, 0.1);
                    border: 1px solid rgba(81, 207, 102, 0.3);
                    border-radius: 6px;
                    font-weight: 400;
                """)
                self.result_label.parent().setStyleSheet("""
                    QFrame {
                        background: rgba(81, 207, 102, 0.05);
                        border-radius: 10px;
                        border: 1px solid rgba(81, 207, 102, 0.2);
                    }
                """)
                info_text = "POSITIVE: AI model predicts benign case.\n\nContinue regular health screenings.\nMaintain healthy lifestyle habits.\nMonitor any symptom changes.\nFollow up with healthcare provider.\n\nPrediction indicates low cancer risk based on cellular analysis."
                
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
            self.confidence_label.setStyleSheet("color: #cccccc; font-weight: 400; background: transparent; border: none;")
            self.info_text.setPlainText(info_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An error occurred during prediction: {str(e)}")
            
    def clear_inputs(self):
        for input_field in self.feature_inputs.values():
            input_field.clear()
        self.result_label.setText("Ready")
        self.result_label.setStyleSheet("color: #aaaaaa; background: transparent; border: none; font-weight: 300;")
        self.result_label.parent().setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        self.confidence_label.setText("Awaiting input")
        self.confidence_label.setStyleSheet("color: #777777; background: transparent; border: none; font-weight: 300;")
        self.info_text.setPlainText("Enter patient measurements and click Analyze for AI-powered diagnosis.\n\nUse Generate for random test data or Sample for known case.")
        
    def generate_random_data(self):
        """Generate realistic breast cancer measurement data based on seed"""
        try:
            seed_text = self.seed_input.text().strip()
            if not seed_text:
                # Generate random seed if none provided
                seed = random.randint(1, 999999)
                self.seed_input.setText(str(seed))
            else:
                try:
                    seed = int(seed_text)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Seed", "Please enter a valid integer seed.")
                    return
            
            # Set random seed for reproducible results
            np.random.seed(seed)
            random.seed(seed)
            
            # Realistic ranges for breast cancer measurements based on medical literature
            feature_ranges = {
                # Mean measurements
                "mean_radius": (6.0, 28.0),
                "mean_texture": (9.0, 40.0), 
                "mean_perimeter": (43.0, 189.0),
                "mean_area": (143.0, 2501.0),
                "mean_smoothness": (0.05, 0.16),
                "mean_compactness": (0.02, 0.35),
                "mean_concavity": (0.0, 0.43),
                "mean_concave_points": (0.0, 0.20),
                "mean_symmetry": (0.11, 0.30),
                "mean_fractal_dimension": (0.05, 0.10),
                
                # Standard error measurements
                "se_radius": (0.1, 2.9),
                "se_texture": (0.4, 4.9),
                "se_perimeter": (0.8, 22.0),
                "se_area": (6.0, 542.0),
                "se_smoothness": (0.002, 0.031),
                "se_compactness": (0.002, 0.135),
                "se_concavity": (0.0, 0.40),
                "se_concave_points": (0.0, 0.053),
                "se_symmetry": (0.008, 0.079),
                "se_fractal_dimension": (0.001, 0.030),
                
                # Worst measurements
                "worst_radius": (7.9, 36.0),
                "worst_texture": (12.0, 49.0),
                "worst_perimeter": (50.0, 251.0),
                "worst_area": (185.0, 4254.0),
                "worst_smoothness": (0.07, 0.22),
                "worst_compactness": (0.03, 1.06),
                "worst_concavity": (0.0, 1.25),
                "worst_concave_points": (0.0, 0.29),
                "worst_symmetry": (0.16, 0.66),
                "worst_fractal_dimension": (0.055, 0.21)
            }
            
            # Generate correlated realistic values
            generated_values = {}
            
            # Generate base radius (affects other measurements)
            base_radius = np.random.uniform(6.0, 28.0)
            generated_values["mean_radius"] = base_radius
            
            # Generate correlated measurements
            for field_name, (min_val, max_val) in feature_ranges.items():
                if field_name == "mean_radius":
                    continue
                    
                if "radius" in field_name:
                    # Radius measurements are correlated
                    if "se_" in field_name:
                        value = base_radius * np.random.uniform(0.01, 0.15)
                    elif "worst_" in field_name:
                        value = base_radius * np.random.uniform(1.1, 1.8)
                    else:
                        value = np.random.uniform(min_val, max_val)
                        
                elif "perimeter" in field_name:
                    # Perimeter correlates with radius (2πr)
                    if "mean_" in field_name:
                        value = 2 * np.pi * base_radius * np.random.uniform(0.9, 1.1)
                    elif "se_" in field_name:
                        value = np.random.uniform(min_val, max_val)
                    else:  # worst
                        value = 2 * np.pi * generated_values.get("worst_radius", base_radius * 1.4) * np.random.uniform(0.9, 1.1)
                        
                elif "area" in field_name:
                    # Area correlates with radius² (πr²)
                    if "mean_" in field_name:
                        value = np.pi * (base_radius ** 2) * np.random.uniform(0.8, 1.2)
                    elif "se_" in field_name:
                        value = np.random.uniform(min_val, max_val)
                    else:  # worst
                        worst_radius = generated_values.get("worst_radius", base_radius * 1.4)
                        value = np.pi * (worst_radius ** 2) * np.random.uniform(0.8, 1.2)
                        
                else:
                    # Other measurements with some correlation to size
                    size_factor = (base_radius - 6.0) / (28.0 - 6.0)  # Normalize to 0-1
                    
                    if "texture" in field_name or "smoothness" in field_name:
                        # Less correlated with size
                        value = np.random.uniform(min_val, max_val)
                    else:
                        # More correlated with size for compactness, concavity, etc.
                        mid_point = (min_val + max_val) / 2
                        range_size = max_val - min_val
                        value = mid_point + (size_factor - 0.5) * range_size * 0.3 + np.random.uniform(-range_size * 0.3, range_size * 0.3)
                
                # Ensure value is within bounds
                value = max(min_val, min(max_val, value))
                generated_values[field_name] = value
            
            # Fill the input fields with generated values
            for field_name, value in generated_values.items():
                if field_name in self.feature_inputs:
                    # Format to appropriate decimal places
                    if value >= 100:
                        formatted_value = f"{value:.1f}"
                    elif value >= 1:
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = f"{value:.5f}"
                    
                    self.feature_inputs[field_name].setText(formatted_value)
            
            # Show success message - minimal
            QMessageBox.information(self, "Generated", 
                                  f"Generated data using seed: {seed}\n\nData is medically plausible and ready for analysis.\nSame seed reproduces identical results.")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Error generating data: {str(e)}")

    def load_sample_data(self):
        # Sample data for testing (known malignant case)
        sample_values = {
            "mean_radius": "17.99", "mean_texture": "10.38", "mean_perimeter": "122.8",
            "mean_area": "1001", "mean_smoothness": "0.1184", "mean_compactness": "0.2776",
            "mean_concavity": "0.3001", "mean_concave_points": "0.1471", "mean_symmetry": "0.2419",
            "mean_fractal_dimension": "0.07871", "se_radius": "1.095", "se_texture": "0.9053",
            "se_perimeter": "8.589", "se_area": "153.4", "se_smoothness": "0.006399",
            "se_compactness": "0.04904", "se_concavity": "0.05373", "se_concave_points": "0.01587",
            "se_symmetry": "0.03003", "se_fractal_dimension": "0.006193", "worst_radius": "25.38",
            "worst_texture": "17.33", "worst_perimeter": "184.6", "worst_area": "2019",
            "worst_smoothness": "0.1622", "worst_compactness": "0.6656", "worst_concavity": "0.7119",
            "worst_concave_points": "0.2654", "worst_symmetry": "0.4601", "worst_fractal_dimension": "0.1189"
        }
        
        for field_name, value in sample_values.items():
            if field_name in self.feature_inputs:
                self.feature_inputs[field_name].setText(value)

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
