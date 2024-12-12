import sys
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QFormLayout, QLineEdit, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Iris Flower Prediction")
        self.setGeometry(100, 100, 400, 200)

        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """)

        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout()

        
        self.button_saisie = QPushButton("Grasp the characteristics of the flower", self)
        self.button_graphique = QPushButton("View charts", self)

        
        layout.addWidget(self.button_saisie)
        layout.addWidget(self.button_graphique)

        self.button_saisie.clicked.connect(self.open_saisie_window)
        self.button_graphique.clicked.connect(self.open_graphique_window)

        self.main_widget.setLayout(layout)

    def open_saisie_window(self):
        self.saisie_window = SaisieWindow()
        self.saisie_window.show()

    def open_graphique_window(self):
        self.graphique_window = GraphiqueWindow()
        self.graphique_window.show()


class SaisieWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Entering characteristics")
        self.setGeometry(100, 100, 500, 400)

       
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e8f5e9;
            }
            QLabel {
                font-size: 14px;
                font-weight: bold;
                text-align: center;
            }
            QLineEdit {
                font-size: 14px;
                padding: 5px;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                background-color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 5px;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """)

        
        self.widget = QWidget()
        self.setCentralWidget(self.widget)

        
        form_layout = QFormLayout()

        self.sepal_length = QLineEdit(self)
        self.sepal_width = QLineEdit(self)
        self.petal_length = QLineEdit(self)
        self.petal_width = QLineEdit(self)

        form_layout.addRow("Sepal length:", self.sepal_length)
        form_layout.addRow("Sepal width:", self.sepal_width)
        form_layout.addRow("Petal length:", self.petal_length)
        form_layout.addRow("Petal width:", self.petal_width)

        
        button_layout = QVBoxLayout()
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setFixedSize(150, 40)  
        self.predict_button.clicked.connect(self.make_prediction)

        self.back_button = QPushButton("Back", self)
        self.back_button.setFixedSize(150, 40)  
        self.back_button.clicked.connect(self.close)

        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.back_button)

        
        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; color: green; margin-top: 20px;")

       
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)   
        main_layout.addLayout(button_layout)  
        main_layout.addWidget(self.result_label) 

        self.widget.setLayout(main_layout)

    def make_prediction(self):
        try:
           
            sepal_length = float(self.sepal_length.text())
            sepal_width = float(self.sepal_width.text())
            petal_length = float(self.petal_length.text())
            petal_width = float(self.petal_width.text())

            
            file_path = ('Iris.csv')
            iris = pd.read_csv(file_path)
            iris = iris.drop(columns=['Id'])

            X = iris.drop(columns=['Species'])
            y = iris['Species']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            
            sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=X.columns)
            prediction = model.predict(sample)

            
            self.result_label.setText(f"flower species : {prediction[0]}")

        except ValueError:
            self.result_label.setText("Please enter valid values.")
            self.result_label.setStyleSheet("font-size: 16px; color: red;")

    def show_prediction(self, prediction):
        
        if hasattr(self, 'error_label'):
            self.error_label.deleteLater()

        
        self.error_label = QLabel(f"Species : {prediction[0]}", self)
        self.error_label.setStyleSheet("font-size: 16px; color: green;")
        self.error_label.move(20, 200)
        self.error_label.resize(400, 50)
        self.error_label.show()


    def show_error(self, message):
        self.error_label = QLabel(message, self)
        self.error_label.move(20, 200)
        self.error_label.show()


class GraphiqueWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        

        self.setStyleSheet("""
            QMainWindow {
                background-color: #fff3e0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """)

    
        self.show_graphics()   

    def show_graphics(self):
        file_path = ('Iris.csv')
        iris = pd.read_csv(file_path)
        iris = iris.drop(columns=['Id'])

        iris_df = pd.DataFrame(iris.drop(columns=['Species']))
        iris_df['species'] = iris['Species']

        
        sns.pairplot(iris_df, hue='species', diag_kind='kde')
        plt.show()

    


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
