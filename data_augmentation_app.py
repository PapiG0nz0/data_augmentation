# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data_augmentation.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_input_path = QtWidgets.QPushButton(self.centralwidget)
        self.btn_input_path.setGeometry(QtCore.QRect(610, 520, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.btn_input_path.setFont(font)
        self.btn_input_path.setObjectName("btn_input_path")
        self.current_path = QtWidgets.QLabel(self.centralwidget)
        self.current_path.setGeometry(QtCore.QRect(10, 10, 171, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.current_path.setFont(font)
        self.current_path.setObjectName("current_path")
        self.btn_output_path = QtWidgets.QPushButton(self.centralwidget)
        self.btn_output_path.setGeometry(QtCore.QRect(610, 570, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.btn_output_path.setFont(font)
        self.btn_output_path.setObjectName("btn_output_path")
        self.output_path = QtWidgets.QLabel(self.centralwidget)
        self.output_path.setGeometry(QtCore.QRect(10, 50, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.output_path.setFont(font)
        self.output_path.setObjectName("output_path")
        self.btn_brightness = QtWidgets.QPushButton(self.centralwidget)
        self.btn_brightness.setGeometry(QtCore.QRect(10, 520, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_brightness.setFont(font)
        self.btn_brightness.setObjectName("btn_brightness")
        self.btn_contrast = QtWidgets.QPushButton(self.centralwidget)
        self.btn_contrast.setGeometry(QtCore.QRect(160, 520, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_contrast.setFont(font)
        self.btn_contrast.setObjectName("btn_contrast")
        self.btn_flip_left_right = QtWidgets.QPushButton(self.centralwidget)
        self.btn_flip_left_right.setGeometry(QtCore.QRect(310, 520, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_flip_left_right.setFont(font)
        self.btn_flip_left_right.setObjectName("btn_flip_left_right")
        self.btn_up_down = QtWidgets.QPushButton(self.centralwidget)
        self.btn_up_down.setGeometry(QtCore.QRect(10, 590, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_up_down.setFont(font)
        self.btn_up_down.setObjectName("btn_up_down")
        self.btn_quality = QtWidgets.QPushButton(self.centralwidget)
        self.btn_quality.setGeometry(QtCore.QRect(310, 590, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_quality.setFont(font)
        self.btn_quality.setObjectName("btn_quality")
        self.btn_hue = QtWidgets.QPushButton(self.centralwidget)
        self.btn_hue.setGeometry(QtCore.QRect(160, 590, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_hue.setFont(font)
        self.btn_hue.setObjectName("btn_hue")
        self.btn_saturation = QtWidgets.QPushButton(self.centralwidget)
        self.btn_saturation.setGeometry(QtCore.QRect(460, 520, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_saturation.setFont(font)
        self.btn_saturation.setObjectName("btn_saturation")
        self.btn_random = QtWidgets.QPushButton(self.centralwidget)
        self.btn_random.setGeometry(QtCore.QRect(460, 590, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_random.setFont(font)
        self.btn_random.setObjectName("btn_random")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 90, 201, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.logs = QtWidgets.QTextBrowser(self.centralwidget)
        self.logs.setGeometry(QtCore.QRect(10, 110, 781, 401))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.logs.setFont(font)
        self.logs.setObjectName("logs")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 791, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_input_path.setText(_translate("MainWindow", "Directorio de entrada"))
        self.current_path.setText(_translate("MainWindow", "Directorio de entrada:"))
        self.btn_output_path.setText(_translate("MainWindow", "Directorio de salida"))
        self.output_path.setText(_translate("MainWindow", "Directorio de salida:"))
        self.btn_brightness.setText(_translate("MainWindow", "Random Brightness"))
        self.btn_contrast.setText(_translate("MainWindow", "Random contrast"))
        self.btn_flip_left_right.setText(_translate("MainWindow", "Flip left to right"))
        self.btn_up_down.setText(_translate("MainWindow", "Up down"))
        self.btn_quality.setText(_translate("MainWindow", "Random Quality"))
        self.btn_hue.setText(_translate("MainWindow", "Random HUE"))
        self.btn_saturation.setText(_translate("MainWindow", "Random saturation"))
        self.btn_random.setText(_translate("MainWindow", "Random"))
        self.label.setText(_translate("MainWindow", "Historial"))
        self.logs.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:14pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())