import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


class ImageClassifierModel:
    def __init__(self):
        self.image = None
        self.model = None

    def load_image(self, image_path):
        """
        Đọc và xử lý ảnh từ đường dẫn. Ảnh sẽ được chuyển thành grayscale, sau đó resize
        và làm phẳng thành một vector.
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Không thể tải ảnh. Đảm bảo đường dẫn đúng.")

            image = cv2.resize(image, (64, 64))  # Resize ảnh cho phù hợp với dữ liệu hoa
            self.image = image.flatten()  # Chuyển ảnh thành vector một chiều
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def load_dataset(self, folder_path):
        """
        Tải dữ liệu từ thư mục ảnh và gán nhãn dựa trên tên thư mục con.
        folder_path: đường dẫn tới thư mục chứa các thư mục con với tên lớp (labels)
        """
        images = []
        labels = []
        class_names = []

        # Duyệt qua các thư mục con (mỗi thư mục là một nhãn)
        for class_name in os.listdir(folder_path):
            class_folder = os.path.join(folder_path, class_name)
            if os.path.isdir(class_folder):
                class_names.append(class_name)  # Thêm nhãn (label) vào danh sách

                # Duyệt qua các ảnh trong từng thư mục con
                for file_name in os.listdir(class_folder):
                    file_path = os.path.join(class_folder, file_name)
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở chế độ grayscale
                    if image is not None:
                        image = cv2.resize(image, (64, 64))  # Resize ảnh về kích thước 64x64
                        images.append(image.flatten())  # Làm phẳng ảnh (flatten)
                        labels.append(class_names.index(class_name))  # Gán nhãn số cho từng lớp

        return np.array(images), np.array(labels), class_names

    def train_model(self, folder_path, classifier_name):
        """
        Huấn luyện mô hình phân loại dựa trên thư mục ảnh.
        folder_path: đường dẫn tới thư mục ảnh
        classifier_name: thuật toán phân loại (SVM, KNN, Decision Tree)
        """
        # Tải dữ liệu và nhãn từ thư mục
        X, y, class_names = self.load_dataset(folder_path)

        # Chia tập dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Chọn mô hình phân loại
        if classifier_name == 'SVM':
            self.model = svm.SVC()
        elif classifier_name == 'KNN':
            self.model = neighbors.KNeighborsClassifier()
        elif classifier_name == 'Decision Tree':
            self.model = tree.DecisionTreeClassifier()

        # Huấn luyện mô hình
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()

        # Dự đoán kết quả trên tập kiểm tra
        y_pred = self.model.predict(X_test)

        # Tính các độ đo so sánh
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        return {
            'time': end_time - start_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_names': class_names
        }

    def classify_image(self):
        """
        Phân loại ảnh đã chọn (nếu đã được tải và xử lý).
        """
        if self.image is None or self.model is None:
            raise ValueError("Ảnh chưa được tải hoặc mô hình chưa được huấn luyện.")

        # Dự đoán ảnh đã chọn
        image_prediction = self.model.predict([self.image])

        return image_prediction[0]
