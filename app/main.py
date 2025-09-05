import sys
from PyQt6.QtWidgets import QWidget, QApplication, QPushButton

app = QApplication(sys.argv)
screen_res = app.primaryScreen().size()
width, height = screen_res.width(), screen_res.height()

class MainWindow(QWidget):
    def __init__(self):

        # Initialize the window
        super().__init__()                                      
        self.setWindowTitle("SVM Classifier for Breast Cancer Prediction")
        self.setGeometry(0, 0, width, height)

        print("Initialized MainWindow")           
        print(f"Window size set to: {width}x{height}")                                  

        # UI Components
        button = QPushButton("Click Me", self)
        button.resize(120, 50)
        button.move((width - button.width()) // 2,
                    (height - button.height()) // 2)
        
        print("Button added to the window")                                                                                       
        
def main ():

    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()  # Show the window maximized after creation

    print("Application started and window shown maximized")
    
    sys.exit(app.exec())
        

if __name__ == "__main__":
    main()
