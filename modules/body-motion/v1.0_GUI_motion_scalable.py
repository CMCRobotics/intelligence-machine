# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QWidget, QComboBox, QPushButton, QSpacerItem, QSizePolicy
from FaceDec_func import process_frame
from pose_estimation_flex import process_pose_frame

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        MainWindow.setStyleSheet("background-color: rgb(202, 235, 255);")

        # --- Central widget ---
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        # === Global Layout ===
        main_layout = QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- Title ---
        self.title_label = QLabel("Motion Capture", self.centralwidget)
        font = QtGui.QFont("Myanmar Text", 24, QtGui.QFont.Bold)
        self.title_label.setFont(font)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.title_label)
        
        # --- Middle Section: Video + BT Box ---
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(20)
        main_layout.addLayout(middle_layout)

        # --- Left side: video widget ---
        self.video_widget = QLabel(self.centralwidget)
        self.video_widget.setStyleSheet("background-color: rgb(202, 235, 255);")
        self.video_widget.setAlignment(QtCore.Qt.AlignCenter) 
        video_layout = QVBoxLayout(self.video_widget)
        video_layout.addStretch()
        middle_layout.addWidget(self.video_widget, 2)

        # --- Right side: two info boxes vertically ---
        info_layout = QVBoxLayout()
        info_layout.setSpacing(20)
        middle_layout.addLayout(info_layout, 1)

        # === BT Box ===
        self.BT_Box = QWidget(self.centralwidget)
        self.BT_Box.setStyleSheet("background-color: rgb(247, 249, 255);")
        bt_layout = QGridLayout(self.BT_Box)
        bt_layout.setContentsMargins(15, 15, 15, 15)
        bt_layout.setVerticalSpacing(10)

        self.BT_label_title = QLabel("Body movements:")
        font_bold = QtGui.QFont()
        font_bold.setPointSize(14)
        font_bold.setBold(True)
        self.BT_label_title.setFont(font_bold)
        bt_layout.addWidget(self.BT_label_title, 0, 0, 1, 2, QtCore.Qt.AlignCenter)

        self.bt_value_labels = {} 

        labels_bt = [("Rise arms:", "###"), ("Flex left arm:", "###"), ("Flex right arm:", "###")]
        for i, (label_text, value_text) in enumerate(labels_bt, start=1):
            lbl = QLabel(label_text)
            lbl.setFont(QtGui.QFont("", 14))
            val = QLabel(value_text)
            val.setFont(QtGui.QFont("", 14))
            bt_layout.addWidget(lbl, i, 0)
            bt_layout.addWidget(val, i, 1, QtCore.Qt.AlignRight)

            self.bt_value_labels[label_text] = val

        info_layout.addWidget(self.BT_Box)
        self.BT_Box.setEnabled(False)

        # === FD Box ===
        self.FD_box = QWidget(self.centralwidget)
        self.FD_box.setStyleSheet("background-color: rgb(247, 249, 255);")
        fd_layout = QGridLayout(self.FD_box)
        fd_layout.setContentsMargins(15, 15, 15, 15)
        fd_layout.setVerticalSpacing(10)

        self.FD_label_title = QLabel("Face movements:")
        self.FD_label_title.setFont(font_bold)
        fd_layout.addWidget(self.FD_label_title, 0, 0, 1, 2, QtCore.Qt.AlignCenter)

        self.fd_value_labels = {} 

        labels_fd = [("Looking Left:", "###"), ("Looking Right:", "###"), ("Looking Down:", "###"), ("Looking Up:", "###"), ("Forward:", "###")]
        for i, (label_text, value_text) in enumerate(labels_fd, start=1):
            lbl = QLabel(label_text)
            lbl.setFont(QtGui.QFont("", 14))
            val = QLabel(value_text)
            val.setFont(QtGui.QFont("", 14))
            fd_layout.addWidget(lbl, i, 0)
            fd_layout.addWidget(val, i, 1, QtCore.Qt.AlignRight)

            self.fd_value_labels[label_text] = val

        info_layout.addWidget(self.FD_box)
        self.FD_box.setEnabled(False)

        # === Bottom Section: ComboBox + Buttons ===
        bottom_layout = QVBoxLayout()
        main_layout.addLayout(bottom_layout)

        # Combo box
        self.exercise_comboBox = QComboBox()
        self.exercise_comboBox.setFont(QtGui.QFont("", 16))
        self.exercise_comboBox.setFixedHeight(50) 
        self.exercise_comboBox.addItems(["Choose movement:", "Body motion", "Face detection"])
        self.exercise_comboBox.model().item(0).setEnabled(False)
        bottom_layout.addWidget(self.exercise_comboBox, alignment=QtCore.Qt.AlignHCenter)

        self.exercise_comboBox.currentIndexChanged.connect(self.on_exercise_changed)

        # Buttons
        btn_layout = QHBoxLayout()
        bottom_layout.addLayout(btn_layout)
        btn_layout.addStretch()

        self.start_pushButton = QPushButton("START")
        self.start_pushButton.setFont(QtGui.QFont("", 20))
        self.start_pushButton.setStyleSheet("background-color: rgb(109, 171, 39); color: white;")
        btn_layout.addWidget(self.start_pushButton)

        self.stop_pushButton = QPushButton("STOP")
        self.stop_pushButton.setFont(QtGui.QFont("", 20))
        self.stop_pushButton.setStyleSheet("background-color: rgb(209, 42, 30); color: white;")
        btn_layout.addWidget(self.stop_pushButton)

        btn_layout.addStretch()

        # --- Status bar ---
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # Inizializza il flusso video e il timer
        self.cap = cv2.VideoCapture(1)  # Usa la webcam predefinita
        self.timer = QtCore.QTimer(self)
        self.start_pushButton.clicked.connect(self.start_video)
        self.stop_pushButton.clicked.connect(self.stop_video)
        self.timer.timeout.connect(self.update_frame)
        #self.timer.start(30)  # Aggiorna il frame ogni 30ms (~33fps)

    def start_video(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
            self.cap.open(1)
           
        # Avvia il timer
        self.timer.start(30)

    def stop_video(self):
        # Ferma il timer
        self.timer.stop()
        # Rilascia la webcam
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        # Pulisci il QLabel
        self.video_widget.clear()

        # --- RESET Face detection counters ---
        from FaceDec_func import counters as fd_counters 
        global previous_pose
        for key in fd_counters.keys():
            fd_counters[key] = 0
            self.fd_value_labels[key + ":"].setText("0")
        previous_pose = "Forward"

        # --- RESET Body motion counters ---
        from pose_estimation_flex import counters as bt_counters 
        for key in bt_counters.keys():
            bt_counters[key] = 0
            self.bt_value_labels[key + ":"].setText("0")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            mode = self.exercise_comboBox.currentText()
            
            if mode == "Face detection":
                frame_to_show, counters  = process_frame(frame)
      
                self.fd_value_labels["Looking Left:"].setText(str(counters["Looking Left"]))
                self.fd_value_labels["Looking Right:"].setText(str(counters["Looking Right"]))
                self.fd_value_labels["Looking Down:"].setText(str(counters["Looking Down"]))
                self.fd_value_labels["Looking Up:"].setText(str(counters["Looking Up"]))
                self.fd_value_labels["Forward:"].setText(str(counters["Forward"]))

            else:
                frame_to_show, counters = process_pose_frame(frame)
                self.bt_value_labels["Flex left arm:"].setText(str(counters["Flex left arm"]))
                self.bt_value_labels["Flex right arm:"].setText(str(counters["Flex right arm"]))
                self.bt_value_labels["Rise arms:"].setText(str(counters["Rise arms"]))

            # Da BGR a RGB
            frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)

            # CQImage per visualizzare il frame in QLabel
            height, width, channels = frame_rgb.shape
            bytes_per_line = channels * width
            qimg = QtGui.QImage(frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

            self.video_widget.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_widget.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def on_exercise_changed(self, index):
        text = self.exercise_comboBox.currentText()

        if text == "Body motion":
            self.BT_Box.setEnabled(True)
            self.FD_box.setEnabled(False)
        elif text == "Face detection":
            self.BT_Box.setEnabled(False)
            self.FD_box.setEnabled(True)
        else:
            # "Choose movement" o altri valori
            self.BT_Box.setEnabled(False)
            self.FD_box.setEnabled(False)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
