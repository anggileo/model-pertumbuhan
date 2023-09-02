import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

class GrowthModelApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Model Pertumbuhan')
        self.setGeometry(100, 100, 800, 500)

        self.layout = QVBoxLayout()

        self.label_growth = QLabel('Masukkan data pertumbuhan (pisahkan dengan koma):')
        self.layout.addWidget(self.label_growth)

        self.data_growth_input = QLineEdit()
        self.layout.addWidget(self.data_growth_input)

        self.label_days = QLabel('Masukkan data hari (pisahkan dengan koma):')
        self.layout.addWidget(self.label_days)

        self.data_days_input = QLineEdit()
        self.layout.addWidget(self.data_days_input)

        self.label_target = QLabel('Masukkan target pertumbuhan:')
        self.layout.addWidget(self.label_target)

        self.target_input = QLineEdit()
        self.layout.addWidget(self.target_input)

        self.load_csv_button = QPushButton('Unggah Data dari CSV')
        self.load_csv_button.clicked.connect(self.load_csv_data)
        self.layout.addWidget(self.load_csv_button)

        self.predict_button_exp = QPushButton('Prediksi Pertumbuhan Eksponensial')
        self.predict_button_exp.clicked.connect(self.predict_exponential_growth)
        self.layout.addWidget(self.predict_button_exp)

        self.predict_button_log = QPushButton('Prediksi Pertumbuhan Logistik')
        self.predict_button_log.clicked.connect(self.predict_logistic_growth)
        self.layout.addWidget(self.predict_button_log)

        # Menambahkan widget plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.result_label = QLabel('')
        self.layout.addWidget(self.result_label)

        # Menambahkan label persamaan pertumbuhan
        self.equation_label = QLabel('Persamaan Pertumbuhan:')
        self.layout.addWidget(self.equation_label)

        self.setLayout(self.layout)

    def exponential_growth(self, x, a, b):
        return a * np.exp(b * x)

    def logistic_growth(self, x, a, b, c):
        return c / (1 + np.exp(-a * (x - b)))

    def load_csv_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Unggah Data CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_name:
            try:
                with open(file_name, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = list(csv_reader)

                    growth_data = [value.strip("'") for value in data[0]]
                    days_data = [value.strip("'") for value in data[1]]

                    self.data_growth_input.setText(','.join(growth_data))
                    self.data_days_input.setText(','.join(days_data))
            except Exception as e:
                self.result_label.setText(f'Error: {str(e)}')

    def predict_exponential_growth(self):
        growth_data_text = self.data_growth_input.text()
        growth_data_list = [float(x.strip()) for x in growth_data_text.split(',')]

        days_data_text = self.data_days_input.text()
        days_data_list = [float(x.strip()) for x in days_data_text.split(',')]

        if len(growth_data_list) != len(days_data_list) or len(growth_data_list) < 2:
            self.result_label.setText('Masukkan minimal dua data pertumbuhan dan hari yang sesuai.')
            return

        days = np.array(days_data_list)
        growth_units = np.array(growth_data_list)

        try:
            params, _ = curve_fit(self.exponential_growth, days, growth_units)
        except RuntimeError:
            self.result_label.setText('Fitting model eksponensial gagal. Pastikan data sesuai dengan model.')
            return

        a, b = params

        target_text = self.target_input.text()
        try:
            target = float(target_text)
        except ValueError:
            self.result_label.setText('Masukkan target yang valid.')
            return

        predicted_days = np.log(target / a) / b

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(days, growth_units, label='Data')
        ax.plot(days, self.exponential_growth(days, a, b), 'r', label='Model Eksponensial')
        ax.set_xlabel('Hari')
        ax.set_ylabel('Unit Pertumbuhan')
        ax.legend()
        ax.set_title('Prediksi Pertumbuhan Eksponensial')
        self.canvas.draw()

        self.result_label.setText(f'Target {target} unit akan tercapai dalam {predicted_days:.2f} hari.')
        equation_label_text = f'Persamaan Pertumbuhan Eksponensial: P(t) = {a:.2f} * e^({b:.2f} * t)'
        self.equation_label.setText(equation_label_text)

    def predict_logistic_growth(self):
        growth_data_text = self.data_growth_input.text()
        growth_data_list = [float(x.strip()) for x in growth_data_text.split(',')]

        days_data_text = self.data_days_input.text()
        days_data_list = [float(x.strip()) for x in days_data_text.split(',')]

        if len(growth_data_list) != len(days_data_list) or len(growth_data_list) < 2:
            self.result_label.setText('Masukkan minimal dua data pertumbuhan dan hari yang sesuai.')
            return

        days = np.array(days_data_list)
        growth_units = np.array(growth_data_list)

        try:
            params, _ = curve_fit(self.logistic_growth, days, growth_units, maxfev=6000)
        except RuntimeError:
            self.result_label.setText('Fitting model logistik gagal. Pastikan data sesuai dengan model.')
            return

        a, b, c = params

        target_text = self.target_input.text()
        try:
            target = float(target_text)
        except ValueError:
            self.result_label.setText('Masukkan target yang valid.')
            return

        predicted_days = np.log((c / target) - 1) / -a + b

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(days, growth_units, label='Data')
        ax.plot(days, self.logistic_growth(days, a, b, c), 'r', label='Model Logistik')
        ax.set_xlabel('Hari')
        ax.set_ylabel('Unit Pertumbuhan')
        ax.legend()
        ax.set_title('Prediksi Pertumbuhan Logistik')
        self.canvas.draw()

        self.result_label.setText(f'Target {target} unit akan tercapai dalam {predicted_days:.2f} hari.')
        equation_label_text = f'Persamaan Pertumbuhan Logistik: P(t) = {c:.2f} / (1 + e^({-a:.2f} * (t - {b:.2f})))'
        self.equation_label.setText(equation_label_text)


def main():
    app = QApplication(sys.argv)
    window = GrowthModelApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

