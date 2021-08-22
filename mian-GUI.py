# coding=utf-8
import cv2

import os, shutil
#
import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import load_model
import numpy as np
import sys

font = cv2.FONT_HERSHEY_SIMPLEX
from keras.optimizers import Adam
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import utils
from scipy import misc


CLASSES = (
'NORMAL','PNEUMONIA')
#classification
model = load_model('model/model-ResNet50-final.h5')
#Load the trained model and modify the name of the model to load a different model.
# There are two models under the model.


#Initialize the visualization interface.
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        #Initialize the parameters of the interface.
        # self.face_recognition = face.Recognition()
        self.timer_camera = QtCore.QTimer()#timer
        self.timer_camera_capture = QtCore.QTimer()#timer
        self.cap = cv2.VideoCapture()#Open the parameters of the camera
        self.CAM_NUM = 0
        self.set_ui()#initialize the inferface
        self.slot_init()
        self.__flag_work = 0
        self.x = 0

    def set_ui(self):
        #initialize the buttons of the interface
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        #the open button
        self.pushButton = QtWidgets.QPushButton(u'Open the Image')
        # self.addface = QtWidgets.QPushButton(u'create the database')
        # self.captureface = QtWidgets.QPushButton(u'collect')
        # self.saveface = QtWidgets.QPushButton(u'save')
        #the size of the picture
        self.pushButton.setMinimumHeight(50)
        # self.addface.setMinimumHeight(50)
        # self.captureface.setMinimumHeight(50)
        # self.saveface.setMinimumHeight(50)
        #edit box location
        self.lineEdit = QtWidgets.QLineEdit(self)  # create QLineEdit
        # self.lineEdit.textChanged.connect(self.text_changed)
        #edit box size
        self.lineEdit.setMinimumHeight(50)
        #edit box location
        # self.opencamera.move(10, 30)
        # self.captureface.move(10, 50)
        self.lineEdit.move(15, 350)

        # show the information
        #loading tools
        self.label = QtWidgets.QLabel()
        # self.label_move = QtWidgets.QLabel()
        #set up the tools' size
        self.lineEdit.setFixedSize(100, 30)
        #set up sizes of the pictures
        self.label.setFixedSize(641, 481)
        self.label.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.pushButton)
        # self.__layout_fun_button.addWidget(self.addface)
        # self.__layout_fun_button.addWidget(self.captureface)
        # self.__layout_fun_button.addWidget(self.saveface)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label)

        self.setLayout(self.__layout_main)
        # self.label_move.raise_()
        self.setWindowTitle(u'HeartLab_Test')

    def slot_init(self):

        self.pushButton.clicked.connect(self.button_open_image_click)
        # self.addface.clicked.connect(self.button_add_face_click)
        # self.timer_camera.timeout.connect(self.show_camera)
        # self.timer_camera_capture.timeout.connect(self.capture_camera)
        # self.captureface.clicked.connect(self.button_capture_face_click)
        # self.saveface.clicked.connect(self.save_face_click)
        #set up the button function
    def button_open_image_click(self):
        #clear the interface
        self.label.clear()
        #clear the comments
        self.lineEdit.clear()
        #open the image
        imgName, imgType = QFileDialog.getOpenFileName(self, "Open the Image", "", "*.jpg;;*.png;;All Files(*)")
        #get the paths of images
        self.img = misc.imread(os.path.expanduser(imgName), mode='RGB')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # self.detection = self.img
        #resize the picture to the specified size
        self.img = cv2.resize(self.img, (640, 480), interpolation=cv2.INTER_AREA)
        #image is none or not
        if self.img is None:
            return None
        #pre-processing
        code = utils.ImageEncode(imgName)
        #prediction
        ret = model.predict(code)
        print(ret)
        #the category with the greatest similarity
        res1 = np.argmax(ret[0, :])
        #print the category with the greatest similarity
        print('result:', CLASSES[res1])
        #show the category with the greatest similarity on the images
        print ('max:',np.max(ret[0, :]))
        cv2.putText(self.img, str(float('%.2f' % np.max(ret[0, :])) * 100) + '%', (1, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                    thickness=2, lineType=2)
        #show the category with the greatest similarity on the images
        cv2.putText(self.img, str(CLASSES[res1]), (1, 160),
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                  thickness=2, lineType=2)
        #chage the color tube
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2BGRA)
        #transfer the image format
        self.QtImg = QtGui.QImage(self.img_rgb.data, self.img_rgb.shape[1], self.img_rgb.shape[0],
                                  QtGui.QImage.Format_RGB32)
        # show the image in the box;
        # self.label.resize(QtCore.QSize(self.img_rgb.shape[1], self.img_rgb.shape[0]))
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        print(CLASSES[res1])
        #edit box types
        self.lineEdit.setText(CLASSES[res1])
    def closeEvent(self, event):
        #Close button funtion
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()
        #close or not
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"Close", u"Close or Not?")
        #click and confirm
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'Confirm')
        cacel.setText(u'Cancel')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            event.accept()


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
