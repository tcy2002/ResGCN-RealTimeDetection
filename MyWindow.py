# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from sys import exit, argv
import threading
import cv2
import cv_viewer.tracking_viewer as cv_viewer
import time
import model as md
from numpy import random, zeros
import pyzed.sl as sl
import torch
import os


# 随机种子
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# 为主循环创建实体
bodies = sl.Objects()
image = sl.Mat()
data75 = zeros((3, 300, 15, 2))
# 相机状态
is_alive = False
# 主程序状态
is_dead = False
# 是否显示结果


# 显示结果
value75 = 0
out_label75 = 0

class MyWindow(QMainWindow):  # 在其他窗口调用必须更换继承类
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)  # 且需要加入程序入口

    def setupUi(self, MainWindow):
        self.desktop = QApplication.desktop()
        # 获取显示器分辨率大小
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.width, self.height)  # 设置铺满
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 100, self.width - 600, self.height - 450))
        self.label.setObjectName("label")

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(50, self.height - 285, self.width - 600, 160))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(self.width - 400, 250, 300, self.height - 600))
        self.textEdit_2.setObjectName("textEdit_2")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(self.width - 250, 100, 150, 35))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(self.width - 250, 150, 150, 35))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(self.width - 400, 150, 100, 35))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_min = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_min.setGeometry(QtCore.QRect(self.width - 120, 20, 20, 20))
        self.pushButton_min.setObjectName("pushButton_3")
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close.setGeometry(QtCore.QRect(self.width - 70, 20, 20, 20))
        self.pushButton_close.setObjectName("pushButton_3")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(self.width - 400, 100, 100, 35))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(["纯净模式", "存储模式"])  # 设置数据源，纯净模式表示不保存信息
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 30, 300, 50))
        self.label_2.setObjectName("label_2")
        font = QtGui.QFont()
        font.setFamily("华文行楷")
        font.setPointSize(20)  # 设置字体
        self.label_2.setFont(font)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(self.width - 400, 200, 300, 50))
        self.label_3.setObjectName("label_3")
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(15)  # 设置字体
        self.label_3.setFont(font)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(50, self.height - 330, 200, 50))
        self.label_4.setObjectName("label_4")
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(15)  # 设置字体
        self.label_4.setFont(font)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(900, 30, 500, 50))
        self.label_5.setObjectName("label_5")
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(18)  # 设置字体
        self.label_5.setFont(font)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, self.width, self.height))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu.setTitle("选项")
        self.action1 = QtWidgets.QAction(MainWindow)
        self.action1.setEnabled(True)  # 添加新建菜单并设置菜单可用
        self.action1.setShortcutContext(QtCore.Qt.WindowShortcut)
        # self.action1.setIconVisibleInMenu(True)  # 设置图标可见
        self.action1.setObjectName("action1")
        self.action1.setText("帮助")  # 设置菜单文本
        self.action1.setToolTip("帮助")  # 设置提示文本
        # self.action1.setShortcut("Ctrl+A")  # 设置快捷键
        self.action1.triggered.connect(self.actionAbout)  # 只针对特定项绑定
        self.menu.addAction(self.action1)  # 加入菜单1
        self.menubar.addAction(self.menu.menuAction())  # 将菜单添加
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.QSS()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.openCamera)
        self.pushButton_2.clicked.connect(self.closeCamera)
        self.pushButton_3.clicked.connect(self.clearinfo)
        self.pushButton_min.clicked.connect(self.showMinimized)
        self.pushButton_close.clicked.connect(self.queryExit)
        self.timer = QTimer()
        self.timer.timeout.connect(self.showtime)  # 这个通过调用槽函数来刷新时间
        self.timer.start(1000)  # 1s刷新一次

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "异常行为捕捉系统"))
        self.showMaximized()  # 最大化
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 无边框
        self.label.setPixmap(QtGui.QPixmap("image/tracker.png"))
        self.label.setScaledContents(True)  # 图片适应label大小
        self.pushButton.setText(_translate("MainWindow", "开启相机"))
        self.pushButton_2.setText(_translate("MainWindow", "关闭相机"))
        self.pushButton_3.setText(_translate("MainWindow", "清空输出"))
        self.label_2.setText(_translate("MainWindow", "异常行为捕捉系统"))
        self.label_3.setText(_translate("MainWindow", "系统信息："))
        self.label_4.setText(_translate("MainWindow", "检测信息："))
        time = QDateTime.currentDateTime()  # 获取当前时间
        timedisplay = time.toString("yyyy-MM-dd hh:mm:ss dddd")  # 格式化一下时间
        self.label_5.setText(_translate("MainWindow", timedisplay))
        self.statusbar.showMessage("未开始检测", 0)  # 第二个参数为要显示的时间（毫秒为单位），为0则一直显示
        self.comboBox.currentIndexChanged.connect(self.modeChange)
        self.mode = self.comboBox.currentText()

        self.textEdit_2.setText(time.toString("yyyy-MM-dd hh:mm:ss") +
                                "\n" + "  系统启动：当前模式为——" + self.comboBox.currentText() + "\n")

    def QSS(self):
        self.setStyleSheet('''QMainWindow{background-color:#576690;}''')
        self.menubar.setStyleSheet('''QMenuBar{background-color:#495579;color:white;}''')
        self.statusbar.setStyleSheet('''QStatusBar{background-color:#60709f;color:white;}''')
        self.menu.setStyleSheet('''QMenu{background-color:#07021d;color:white;}''')
        self.pushButton_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.pushButton_min.setStyleSheet(
            '''QPushButton{background:#FFFF00;border-radius:5px;}QPushButton:hover{background:#FFD700;}''')
        self.pushButton.setStyleSheet(
            '''QPushButton{background:#ADD8E6;border-radius:10px;}QPushButton:hover{background:#00F5FF;}''')
        self.pushButton_2.setStyleSheet(
            '''QPushButton{background:#ADD8E6;border-radius:10px;}QPushButton:hover{background:#00F5FF;}''')
        self.pushButton_3.setStyleSheet(
            '''QPushButton{background:#ADD8E6;border-radius:10px;}QPushButton:hover{background:#00F5FF;}''')
        self.comboBox.setStyleSheet('''QComboBox{background:#F5FFFA;border-radius:8px;}''')
        self.textEdit.setStyleSheet('''QTextEdit{background:#686f82;border-radius:15px;color:white}''')
        self.textEdit_2.setStyleSheet('''QTextEdit{background:#686f82;border-radius:15px;color:white}''')

    def queryExit(self):
        global is_alive, is_dead
        res = QtWidgets.QMessageBox.question(self, "关闭", "是否退出系统?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        if res == QtWidgets.QMessageBox.Yes:
            if is_alive:  # 如果相机还未关闭
                is_alive = False
                time.sleep(0.05)
            is_dead = True
            QtCore.QCoreApplication.instance().exit()

    def modeChange(self):
        self.mode = self.comboBox.currentText()
        time = QDateTime.currentDateTime()
        self.textEdit_2.setText(self.textEdit_2.toPlainText() + time.toString("yyyy-MM-dd hh:mm:ss") +
                                "\n" + "  改变模式:当前模式为——" + self.comboBox.currentText() + "\n")

    def mode_operate(self, img, info):
        if self.mode == "纯净模式":
            pass
        else:
            if not os.path.exists(os.getcwd() + "/result"):
                os.mkdir(os.getcwd() + "/result")  # 创建一个结果文件夹
                os.mkdir(os.getcwd() + "/result/img_result")  # 图片文件夹
            img_path = os.getcwd() + "/result/img_result/"

            cv2.imwrite(img_path + info + ".jpg", img)
            f = open(os.getcwd() + "/result/info_result.txt", "a")
            time = QDateTime.currentDateTime()
            f.write(time.toString("yyyy-MM-dd hh:mm:ss") + "    " + info + "\n")
            f.close()
            self.textEdit_2.setText(self.textEdit_2.toPlainText() + time.toString("yyyy-MM-dd hh:mm:ss") +
                                    "\n" + "  保存成功！" + "\n")

    def clearinfo(self):
        self.textEdit.setText("")
        time = QDateTime.currentDateTime()
        self.textEdit_2.setText(self.textEdit_2.toPlainText() + time.toString("yyyy-MM-dd hh:mm:ss") +
                                "\n" + "  清空输出信息" + "\n")

    def showtime(self):
        time = QDateTime.currentDateTime()  # 获取当前时间
        timedisplay = time.toString("yyyy-MM-dd hh:mm:ss dddd")  # 格式化一下时间
        self.label_5.setText(timedisplay)

    def actionAbout(self):
        QMessageBox.about(None, "帮助", "本界面接口适用于ZED深度相机\n纯净模式：对视频内容与检测内容仅显示而不保存\n存储模式："
                                      "对检测到的视频图像和检测内容进行储存")

    def show_camera(self):

        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (480, 320))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(showImage))

    def openCamera(self):
        global is_alive
        if not is_alive:
            is_alive = True
            self.statusbar.showMessage("检测中...", 0)
            time = QDateTime.currentDateTime()
            self.textEdit_2.setText(self.textEdit_2.toPlainText() + time.toString("yyyy-MM-dd hh:mm:ss") +
                                    "\n" + "  开启检测" + "\n")
            self.camera()
        else:
            QMessageBox.critical(None, "错误", "摄像头已打开")

    # 关闭摄像头
    def closeCamera(self):
        global is_alive
        if is_alive:
            is_alive = False
            time.sleep(0.05)
            self.label.setPixmap(QtGui.QPixmap("image/tracker.png"))
            self.statusbar.showMessage("未检测...", 0)
            time1 = QDateTime.currentDateTime()
            self.textEdit_2.setText(self.textEdit_2.toPlainText() + time1.toString("yyyy-MM-dd hh:mm:ss") +
                                    "\n" + "  关闭检测" + "\n")
        else:
            QMessageBox.critical(None, "错误", "还未打开摄像头")

    def init_zed(self):
        # 相机初始化参数
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # positional_tracking_parameters.set_as_static = True  # 如果相机为静态则使用
        obj_param = sl.ObjectDetectionParameters()
        obj_param.enable_body_fitting = True  # Smooth skeleton move
        obj_param.enable_tracking = True  # Track people across images flow
        obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_param.detection_confidence_threshold = 40
        return init_params, positional_tracking_parameters, obj_param, obj_runtime_param

    # 摄像头主程序
    def camera(self):
        global data75
        # 相机初始化
        zed = sl.Camera()
        init_params, positional_tracking_parameters, obj_param, obj_runtime_param = self.init_zed()

        # 打开摄像头
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)
        zed.enable_positional_tracking(positional_tracking_parameters)
        zed.enable_object_detection(obj_param)
        camera_info = zed.get_camera_information()

        # 显示设置
        display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                           min(camera_info.camera_resolution.height, 720))
        image_scale = [display_resolution.width / camera_info.camera_resolution.width
            , display_resolution.height / camera_info.camera_resolution.height]

        # 主循环显示图像
        start = cv2.getTickCount()
        fps = 0
        key = 0
        while is_alive:
            # 控制帧率（仅针对文字显示）
            end = cv2.getTickCount()
            key = (key + 10) % 60

            # 抓取图像
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # 检索左目
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                # 检索人物对象
                zed.retrieve_objects(bodies, obj_runtime_param)
                # 跟踪显示
                image_left_ocv = image.get_data()
                cv_viewer.render_2D(image_left_ocv, image_scale, bodies.object_list, obj_param.enable_tracking)
                self.show = cv2.resize(image_left_ocv, (480, 320))
                data75 = md.data_process(75, data75, bodies.object_list)

                # 显示帧率
                if not key:
                    fps = cv2.getTickFrequency() / (end - start)
                cv2.putText(self.show, '%.2f' % fps + 'FPS', (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 200, 255))
                start = end

                # 显示检测结果
                if value75 > 3.5 and out_label75 in (50, 51, 52, 55):
                    time = QDateTime.currentDateTime()  # 获取当前时间
                    self.textEdit.setText(
                        f'{md.interpreter[out_label75]}' + "\t" + time.toString("yyyy-MM-dd hh:mm:ss") + "\n")  # str输出信息
                    self.textEdit_2.setText(self.textEdit_2.toPlainText() + time.toString("yyyy-MM-dd hh:mm:ss") +
                                            "\n" + "  检测到异常行为！" + "\n")
                    self.mode_operate(self.show, f'{md.interpreter[out_label75]}')

                self.show = cv2.cvtColor(self.show, cv2.COLOR_BGR2RGB)
                showImage = QImage(self.show.data, self.show.shape[1], self.show.shape[0], QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(showImage))

                cv2.waitKey(1)

        # 退出主循环后关闭摄像头释放内存
        image.free(sl.MEM.CPU)
        zed.disable_object_detection()
        zed.disable_positional_tracking()
        zed.close()

def recognize75():
    global value75, out_label75, data75
    model = md.init_model()
    while not is_dead:
        # 控制帧率（仅针对识别频率）
        if is_alive:
            # 喂入数据
            input_data = md.multi_input(data75, md.connect_joint)
            input_data = torch.Tensor([input_data, ])
            out, _ = model(input_data)
            value75, out_label75 = float(out.max(1)[0]), int(out.max(1)[1])


if __name__ == '__main__':
    app = QtWidgets.QApplication(argv)  # 必须在实例化对象前建立
    mywindow = MyWindow()  # 需要先实例化，然后再从登录窗口内打开
    rec75 = threading.Thread(target=recognize75)
    rec75.start()
    mywindow.show()
    exit(app.exec_())
