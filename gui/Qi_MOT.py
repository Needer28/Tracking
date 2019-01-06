import sys
from PyQt5 import QtWidgets
from Qi_MOTUI import Ui_Form
from pprint import pprint
from PyQt5.QtCore import QBasicTimer
from PyQt5.QtWidgets import QApplication,QMainWindow,QGridLayout,QTabWidget,QPushButton,\
							QVBoxLayout,QHBoxLayout,QWidget,QGraphicsView,QGraphicsScene,\
							QGraphicsPixmapItem,QTabBar
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaMetaData, QMediaPlayer,QMediaContent
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import QUrl
import os
from PyQt5 import QtTest



class Qi_MOT(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(Qi_MOT,self).__init__()
        self.setupUi(self)
        self.progressBar1.setProperty("value", 0)
        self.progressBar2.setProperty("value", 0)
        self.progressBar3.setProperty("value", 0)

        self.timer = QBasicTimer()



    #实现pushButton_click()函数，textEdit是我们放上去的文本框的id
    # def pushButton_click(self):
    #     self.textEdit.setText("你点击了按钮")

    # Here have many functions
    #%%


    # 启动后台程序，在此，可以将其分开，根据选项执行不同的脚本，从而完成调用，方便展示
    #def startProgress1(self):
    #    self.step1 = 0
     #   while self.step1 < 100:
      #      self.step1 = self.step1 + 0.0001
       #     self.progressBar1.setValue(self.step1)

        ## run code command though button      监控场景下的数据再重新调用文件。
        #strfile = ('python /home/qi/projects/LSTM_Qi_v27/py/shiyan5.py')
        ## strfile = ("python /home/qi/projects/testcode/subcode.py")

        #os.chdir("/home/qi/projects/LSTM_Qi_v27/py")
        #p = os.system(strfile)
        #print(p)

    def startProgress1(self):
        self.step1 = 0
        while self.step1 < 100:
            self.step1 = self.step1 + 0.0001
            self.progressBar1.setValue(self.step1)


    def startProgress2(self):
        self.step2 = 0
        while self.step2 < 100:
            self.step2 = self.step2 + 0.0001
            self.progressBar2.setValue(self.step2)

    def startProgress3(self):
        self.step3 = 0
        while self.step3 < 100:
            self.step3 = self.step3 + 0.0001
            self.progressBar3.setValue(self.step3)

    def oridatashow(self):
        if self.orib2.isChecked():
            # self.evatextBrowser.setText("你点击了按钮orib2")
            self.origraphicsView.hide()

            # imgpath = '/home/qi/benchmark/MOT17/train/MOT17-02-SDP/img1/000001.jpg'
            # pix = QPixmap(imgpath)
            # self.orilabel.setPixmap(pix)
            # self.orilabel.setStyleSheet("border: 2px solid red")
            # self.orilabel.setScaledContents(True)

            l = 600
            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-02-SDP'
                path = '/home/qi/benchmark/MOT17/train/%s/img1' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.orilabel.setPixmap(pix)
                self.orilabel.setStyleSheet("border: 2px solid red")
                self.orilabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)


        if self.orib4.isChecked():
            # self.evatextBrowser.setText("你点击了按钮orib4")
            self.origraphicsView.hide()

            l = 1050
            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-04-SDP'
                path = '/home/qi/benchmark/MOT17/train/%s/img1' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.orilabel.setPixmap(pix)
                self.orilabel.setStyleSheet("border: 2px solid red")
                self.orilabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)


        if self.orib5.isChecked():
            # self.evatextBrowser.setText("你点击了按钮orib5")
            self.origraphicsView.hide()
            l = 837

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-05-SDP'
                path = '/home/qi/benchmark/MOT17/train/%s/img1' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.orilabel.setPixmap(pix)
                self.orilabel.setStyleSheet("border: 2px solid red")
                self.orilabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.orib9.isChecked():
            # self.evatextBrowser.setText("你点击了按钮orib9")
            self.origraphicsView.hide()
            l = 525

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-09-SDP'
                path = '/home/qi/benchmark/MOT17/train/%s/img1' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.orilabel.setPixmap(pix)
                self.orilabel.setStyleSheet("border: 2px solid red")
                self.orilabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.orib10.isChecked():
            # self.evatextBrowser.setText("你点击了按钮orib10")
            self.origraphicsView.hide()
            l = 654

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-10-SDP'
                path = '/home/qi/benchmark/MOT17/train/%s/img1' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.orilabel.setPixmap(pix)
                self.orilabel.setStyleSheet("border: 2px solid red")
                self.orilabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.orib11.isChecked():
            # self.evatextBrowser.setText("你点击了按钮orib11")
            self.origraphicsView.hide()
            l = 900

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-11-SDP'
                path = '/home/qi/benchmark/MOT17/train/%s/img1' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.orilabel.setPixmap(pix)
                self.orilabel.setStyleSheet("border: 2px solid red")
                self.orilabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.orib13.isChecked():
            # self.evatextBrowser.setText("你点击了按钮orib13")
            self.origraphicsView.hide()
            l = 750

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-13-SDP'
                path = '/home/qi/benchmark/MOT17/train/%s/img1' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.orilabel.setPixmap(pix)
                self.orilabel.setStyleSheet("border: 2px solid red")
                self.orilabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.oriother.isChecked():
            # self.evatextBrowser.setText("你点击了按钮oriother")
            length = 10
            self.origraphicsView.hide()
            l = 100

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-02'
                path = '/home/qi/benchmark/MOT17/detandtracking/%s/img1' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.orilabel.setPixmap(pix)
                self.orilabel.setStyleSheet("border: 2px solid red")
                self.orilabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)




    def newdatashow(self):
        if self.newb2.isChecked():
            # self.evatextBrowser.setText("你点击了按钮newb2")

            # self.newgraphicsView.hide()
            # imgpath = '/home/qi/benchmark/MOT17/train/MOT17-02-SDP/img1/000001.jpg'
            # pix = QPixmap(imgpath)
            # self.newlabel.setPixmap(pix)
            # self.newlabel.setStyleSheet("border: 2px solid red")
            # self.newlabel.setScaledContents(True)
            # moviepath = '/home/qi/Videos/1.gif'  # gif right
            # movie = QtGui.QMovie(moviepath)
            # movie.setSpeed(100)
            # self.newlabel.setMovie(movie)
            # movie.start()

            self.newgraphicsView.hide()

            l = 600
            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-02'
                path = '/home/qi/benchmark/MOT17/tracking/%s/img2' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.newlabel.setPixmap(pix)
                self.newlabel.setStyleSheet("border: 2px solid red")
                self.newlabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)


        if self.newb4.isChecked():
            # self.evatextBrowser.setText("你点击了按钮newb4")
            self.newgraphicsView.hide()

            l = 1050
            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-04'
                path = '/home/qi/benchmark/MOT17/tracking/%s/img2' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.newlabel.setPixmap(pix)
                self.newlabel.setStyleSheet("border: 2px solid red")
                self.newlabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.newb5.isChecked():
            # self.evatextBrowser.setText("你点击了按钮newb5")
            self.newgraphicsView.hide()
            l = 837

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-05'
                path = '/home/qi/benchmark/MOT17/tracking/%s/img2' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.newlabel.setPixmap(pix)
                self.newlabel.setStyleSheet("border: 2px solid red")
                self.newlabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.newb9.isChecked():
            # self.evatextBrowser.setText("你点击了按钮newb9")
            self.newgraphicsView.hide()
            l = 525

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-09'
                path = '/home/qi/benchmark/MOT17/tracking/%s/img2' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.newlabel.setPixmap(pix)
                self.newlabel.setStyleSheet("border: 2px solid red")
                self.newlabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.newb10.isChecked():
            # self.evatextBrowser.setText("你点击了按钮newb10")
            self.newgraphicsView.hide()
            l = 654

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-10'
                path = '/home/qi/benchmark/MOT17/tracking/%s/img2' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.newlabel.setPixmap(pix)
                self.newlabel.setStyleSheet("border: 2px solid red")
                self.newlabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.newb11.isChecked():
            # self.evatextBrowser.setText("你点击了按钮newb11")
            self.newgraphicsView.hide()
            l = 900

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-11'
                path = '/home/qi/benchmark/MOT17/tracking/%s/img2' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.newlabel.setPixmap(pix)
                self.newlabel.setStyleSheet("border: 2px solid red")
                self.newlabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.newb13.isChecked():
            # self.evatextBrowser.setText("你点击了按钮newb13")
            self.newgraphicsView.hide()
            l = 750

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-13'
                path = '/home/qi/benchmark/MOT17/tracking/%s/img2' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.newlabel.setPixmap(pix)
                self.newlabel.setStyleSheet("border: 2px solid red")
                self.newlabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

        if self.newother.isChecked():
            # self.evatextBrowser.setText("你点击了按钮newother")
            self.newgraphicsView.hide()
            l = 100

            im_names = []
            msecs = 10

            for num in range(1, l + 1):
                if num < 10 and num > 0:
                    pic = '00000' + str(num) + '.jpg'
                elif num < 100:
                    pic = '0000' + str(num) + '.jpg'
                elif num < 1000:
                    pic = '000' + str(num) + '.jpg'
                else:
                    pic = '00' + str(num) + '.jpg'
                im_names.append(pic)

            for im_name in im_names:
                folder = 'MOT17-02'
                path = '/home/qi/benchmark/MOT17/detandtracking/%s/img2' % folder
                imgpath = os.path.join(path, im_name)

                pix = QPixmap(imgpath)
                self.newlabel.setPixmap(pix)
                self.newlabel.setStyleSheet("border: 2px solid red")
                self.newlabel.setScaledContents(True)
                QtTest.QTest.qWait(msecs)

# then we read tracking results pictures

    def evaluation(self):

        project = 'LSTM_Qi_v27'
        trackingUIresults ='trackingUIresults'

        if self.evab2.isChecked():
            # self.evatextBrowser.setText("你点击了按钮evab2")

            filenames = '/home/qi/projects/%s/shiyan/trackingUIresults/README2.md' % project
            f = open(filenames, 'r')
            with f:
                data = f.read()
                self.evatextBrowser.setText(data)


        if self.evab4.isChecked():
            # self.evatextBrowser.setText("你点击了按钮evab4")
            # filenames = '/home/qi/projects/LSTM_Qi_v27/shiyan/README.md'
            filenames = '/home/qi/projects/%s/shiyan/trackingUIresults/README4.md' % project
            f = open(filenames, 'r')
            with f:
                data = f.read()
                self.evatextBrowser.setText(data)

        if self.evab5.isChecked():
            # self.evatextBrowser.setText("你点击了按钮evab5")
            # filenames = '/home/qi/projects/LSTM_Qi_v27/shiyan/README.md'
            filenames = '/home/qi/projects/%s/shiyan/trackingUIresults/README5.md' % project
            f = open(filenames, 'r')
            with f:
                data = f.read()
                self.evatextBrowser.setText(data)

        if self.evab9.isChecked():
            # self.evatextBrowser.setText("你点击了按钮evab9")
            # filenames = '/home/qi/projects/LSTM_Qi_v27/shiyan/README.md'
            filenames = '/home/qi/projects/%s/shiyan/trackingUIresults/README9.md' % project
            f = open(filenames, 'r')
            with f:
                data = f.read()
                self.evatextBrowser.setText(data)

        if self.evab10.isChecked():
            # self.evatextBrowser.setText("你点击了按钮evab10")
            # filenames = '/home/qi/projects/LSTM_Qi_v27/shiyan/README.md'
            filenames = '/home/qi/projects/%s/shiyan/trackingUIresults/README10.md' % project
            f = open(filenames, 'r')
            with f:
                data = f.read()
                self.evatextBrowser.setText(data)

        if self.evab11.isChecked():
            # self.evatextBrowser.setText("你点击了按钮evab11")
            # filenames = '/home/qi/projects/LSTM_Qi_v27/shiyan/README.md'
            filenames = '/home/qi/projects/%s/shiyan/trackingUIresults/README11.md' % project
            f = open(filenames, 'r')
            with f:
                data = f.read()
                self.evatextBrowser.setText(data)

        if self.evab13.isChecked():
            # self.evatextBrowser.setText("你点击了按钮evab13")
            # filenames = '/home/qi/projects/LSTM_Qi_v27/shiyan/README.md'
            filenames = '/home/qi/projects/%s/shiyan/trackingUIresults/README13.md' % project
            f = open(filenames, 'r')
            with f:
                data = f.read()
                self.evatextBrowser.setText(data)

        if self.evaball.isChecked():
            # self.evatextBrowser.setText("您点击了评估所有数据集的按钮！")
            # filenames = '/home/qi/projects/LSTM_Qi_v27/shiyan/README.md'
            filenames = '/home/qi/projects/%s/shiyan/trackingUIresults/READMEall.md' % project
            f = open(filenames, 'r')
            with f:
                data = f.read()
                self.evatextBrowser.setText(data)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = Qi_MOT()
    my_pyqt_form.show()
    sys.exit(app.exec_())