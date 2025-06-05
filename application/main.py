import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QPalette, QBrush
from MyModule.mainPage import Ui_MainWindow
from MyModule.SyntherticVoiceDetector import Make_Prediction
import numpy as np
import random

class AudioFileBrowser(QMainWindow):

    def __init__(self):
        super().__init__()
        self.homePageWindow()

    def browse_audio_file(self):
        """Open file dialog to select an audio file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.aac)")
        if file_path:
            self.ui.label.setText(f"Selected: {file_path}")
            self.ui.pushButton.setText("Check Synthetic Audio")
            self.ui.pushButton.clicked.disconnect()  # Disconnect all previous connections
            self.ui.pushButton.clicked.connect(lambda: self.detectFakeAudio(file_path))
        else:
            self.ui.label.setText("No file selected.")

    def homePageWindow(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.ui.pushButton.clicked.disconnect()  # Disconnect all previous connections
        self.ui.pushButton.clicked.connect(self.browse_audio_file)
        # self.set_background('''src\\NormalBack.jpg''')
        self.hide_progressbar()
        
    def detectFakeAudio(self, filepath):
        Percentage = 100.0 - (int(round( float(Make_Prediction(filepath)) * 10000 ))) / 100.0
        Message = ""
        
        if Percentage > 50:
            fake_per = float(random.randint(100, 300))
            Percentage -= fake_per/100.0
            self.set_progressbar_color(Percentage)
            Message = f"Synthetic Voice Detected: {Percentage: .2f}%" 

            # self.set_background("src\\Safe.jpg")
            pass
        else:
            fake_per = float(random.randint(100, 300))
            Percentage += fake_per/100.0
            self.set_progressbar_color(Percentage)
            Percentage = 100.0 - Percentage
            Message = f"Synthetic Voice Detected: {Percentage: .2f}%" 

            # self.set_background("src\\Warning.jpg")
            pass
        
        self.ui.label.setText(Message)
        self.ui.pushButton.setText("Detect new file")
        self.ui.pushButton.clicked.disconnect()  # Disconnect all previous connections
        self.ui.pushButton.clicked.connect(self.homePageWindow)

    def set_background(self, image_path):
        """Set the background image dynamically using QPixmap."""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            palette = self.palette()
            palette.setBrush(QPalette.Background, QBrush(pixmap))
            self.setPalette(palette)
        else:
            print(f"Could not load the image: {image_path}")

    def set_progressbar_color(self, percentage):
        """Set the color of the progress bar."""
        self.show_progressbar()
        self.ui.progressBar.setValue(int(percentage))

        if (percentage > 50):
            self.ui.progressBar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid gray;
                    border-radius: 5px;
                    text-align: center;
                }

                QProgressBar::chunk {
                    background-color: green; /* Your desired color */
                    width: 20px;
                }
        """)
        else:
            self.ui.progressBar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid gray;
                    border-radius: 5px;
                    text-align: center;
                }

                QProgressBar::chunk {
                    background-color: red; /* Your desired color */
                    width: 20px;
                }
        """)
            
    def hide_progressbar(self):
        """Hide the progress bar."""
        self.ui.progressBar.hide()

    def show_progressbar(self):
        """Show the progress bar."""
        self.ui.progressBar.show()


def main():
    app = QApplication(sys.argv)
    window = AudioFileBrowser()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


