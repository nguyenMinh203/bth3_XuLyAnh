import sys
from PyQt6.QtWidgets import QApplication
from Controller import ImageClassifierController

def main():
    app = QApplication(sys.argv)

    controller = ImageClassifierController()
    controller.view.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
