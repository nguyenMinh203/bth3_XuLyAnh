from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog

class ClassifierView(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.algorithm_label = QLabel("Chọn thuật toán phân loại:")
        layout.addWidget(self.algorithm_label)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["SVM", "KNN", "Decision Tree"])
        layout.addWidget(self.algorithm_combo)

        self.select_image_button = QPushButton("Chọn ảnh")
        layout.addWidget(self.select_image_button)

        self.run_button = QPushButton("Chạy phân loại")
        layout.addWidget(self.run_button)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.setLayout(layout)
        self.setWindowTitle('Image Classifier')
        self.setGeometry(300, 300, 400, 300)

    def open_file_dialog(self):
        # Không cần tạo options trong PyQt6
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.xpm *.jpg)")
        return file_path
