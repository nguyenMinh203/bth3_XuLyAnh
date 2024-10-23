from View import ClassifierView
from Model import ImageClassifierModel

class ImageClassifierController:
    def __init__(self):
        self.model = ImageClassifierModel()
        self.view = ClassifierView()

        # Kết nối sự kiện chọn ảnh và phân loại
        self.view.select_image_button.clicked.connect(self.select_image)
        self.view.run_button.clicked.connect(self.run_classification)

    def select_image(self):
        image_path = self.view.open_file_dialog()
        if image_path:
            self.model.load_image(image_path)
            self.view.result_text.setText(f"Đã chọn ảnh: {image_path}\n")

    def run_classification(self):
        classifier_name = self.view.algorithm_combo.currentText()

        # Huấn luyện mô hình với dữ liệu hoa từ thư mục
        folder_path = "D:/taive/archive/flowers"
        metrics = self.model.train_model(folder_path, classifier_name)

        # Hiển thị các độ đo sau khi huấn luyện
        result_text = (
            f"Thời gian huấn luyện: {metrics['time']:.4f}s\n"
            f"Độ chính xác (Accuracy): {metrics['accuracy']:.4f}\n"
            f"Độ chính xác (Precision): {metrics['precision']:.4f}\n"
            f"Độ nhạy (Recall): {metrics['recall']:.4f}\n"
            f"F1-score: {metrics['f1_score']:.4f}\n"
        )

        self.view.result_text.append(result_text)

        # Phân loại ảnh người dùng đã chọn
        if self.model.image is not None:
            image_prediction = self.model.classify_image()
            class_names = metrics['class_names']
            self.view.result_text.append(f"Dự đoán ảnh: {class_names[image_prediction]}")
        else:
            self.view.result_text.append("Chưa có ảnh được chọn để phân loại.")
