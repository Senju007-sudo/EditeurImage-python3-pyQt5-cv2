# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:34:54 2020

@author: SENJU
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog , QMessageBox
from PyQt5.QtGui import QImage
import cv2
import numpy as np

from fonctions import otsu , griser , egalisationHisto , lisse , etale , luminosite , translation  ,rotation90 , floutter 
from fonctions import inversionCouleur , binarisation , erosion , dilatation , conserver, getLigne , getColonne


class Ui_MainWindow(object):
        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(1242, 698)
            self.size1 = 1200
            self.size2 = 700
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
            MainWindow.setSizePolicy(sizePolicy)
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.label = QtWidgets.QLabel(self.centralwidget)
            self.label.setGeometry(QtCore.QRect(9, 9, 1189, 630))
            self.label.setMaximumSize(QtCore.QSize(1250, 720))
            self.label.setText("")
            self.label.setPixmap(QtGui.QPixmap("../../AIVOPr/images/01972_hsbccelebrationoflight_1920x1080.jpg"))
            self.label.setScaledContents(True)
            self.label.setObjectName("label")
            self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
            self.horizontalSlider.setGeometry(QtCore.QRect(1199, 40, 151, 22))
            self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
            self.horizontalSlider.setObjectName("horizontalSlider")
            self.horizontalSlider_2 = QtWidgets.QSlider(self.centralwidget)
            self.horizontalSlider_2.setGeometry(QtCore.QRect(1200, 90, 160, 22))
            self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
            self.horizontalSlider_2.setObjectName("horizontalSlider_2")
            self.label_2 = QtWidgets.QLabel(self.centralwidget)
            self.label_2.setGeometry(QtCore.QRect(1200, 20, 61, 16))
            self.label_2.setObjectName("label_2")
            self.label_3 = QtWidgets.QLabel(self.centralwidget)
            self.label_3.setGeometry(QtCore.QRect(1200, 70, 61, 16))
            self.label_3.setObjectName("label_3")
            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setEnabled(True)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 1290, 22))
            self.menubar.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
            self.menubar.setDefaultUp(False)
            self.menubar.setNativeMenuBar(True)
            self.menubar.setObjectName("menubar")
            self.menuOuvrir = QtWidgets.QMenu(self.menubar)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("icons/16x16/Open_16px_1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.menuOuvrir.setIcon(icon)
            self.menuOuvrir.setObjectName("menuOuvrir")
            self.menuAmelioration = QtWidgets.QMenu(self.menubar)
            self.menuAmelioration.setObjectName("menuAmelioration")
            self.menuFiltrage = QtWidgets.QMenu(self.menubar)
            self.menuFiltrage.setObjectName("menuFiltrage")
            self.menuBinarisation = QtWidgets.QMenu(self.menubar)
            self.menuBinarisation.setObjectName("menuBinarisation")
            self.menuTransformation = QtWidgets.QMenu(self.menubar)
            self.menuTransformation.setObjectName("menuTransformation")
            self.menuRotation = QtWidgets.QMenu(self.menubar)
            self.menuRotation.setObjectName("menuRotation")
            self.menuTranslation = QtWidgets.QMenu(self.menubar)
            self.menuTranslation.setObjectName("menuTranslation")
            MainWindow.setMenuBar(self.menubar)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)
            self.toolBar = QtWidgets.QToolBar(MainWindow)
            self.toolBar.setObjectName("toolBar")
            MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
            self.toolBar_2 = QtWidgets.QToolBar(MainWindow)
            self.toolBar_2.setObjectName("toolBar_2")
            MainWindow.addToolBar(QtCore.Qt.RightToolBarArea, self.toolBar_2)
            self.actionOuvrir = QtWidgets.QAction(MainWindow)
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap("icons/16x16/Add Image_16px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionOuvrir.setIcon(icon1)
            self.actionOuvrir.setIconVisibleInMenu(False)
            self.actionOuvrir.setObjectName("actionOuvrir")
            self.actionEnregister = QtWidgets.QAction(MainWindow)
            icon2 = QtGui.QIcon()
            icon2.addPixmap(QtGui.QPixmap("icons/16x16/Save_16px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionEnregister.setIcon(icon2)
            self.actionEnregister.setObjectName("actionEnregister")
            self.actionExit = QtWidgets.QAction(MainWindow)
            icon3 = QtGui.QIcon()
            icon3.addPixmap(QtGui.QPixmap("icons/16x16/Exit_16px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionExit.setIcon(icon3)
            self.actionExit.setObjectName("actionExit")
            self.actionEgalisation = QtWidgets.QAction(MainWindow)
            self.actionEgalisation.setObjectName("actionEgalisation")
            self.actionMedian = QtWidgets.QAction(MainWindow)
            self.actionMedian.setObjectName("actionMedian")
            self.actionGaussien = QtWidgets.QAction(MainWindow)
            self.actionGaussien.setObjectName("actionGaussien")
            self.actionMoyenne = QtWidgets.QAction(MainWindow)
            self.actionMoyenne.setObjectName("actionMoyenne")
            self.actionLissage = QtWidgets.QAction(MainWindow)
            self.actionLissage.setObjectName("actionLissage")
            self.actionSeuil = QtWidgets.QAction(MainWindow)
            self.actionSeuil.setObjectName("actionSeuil")
            self.actionErosion = QtWidgets.QAction(MainWindow)
            self.actionErosion.setObjectName("actionErosion")
            self.actionDilatation = QtWidgets.QAction(MainWindow)
            self.actionDilatation.setObjectName("actionDilatation")
            self.actionNiveau_de_gris = QtWidgets.QAction(MainWindow)
            self.actionNiveau_de_gris.setObjectName("actionNiveau_de_gris")
            self.actionNiveau_de_gris_2 = QtWidgets.QAction(MainWindow)
            self.actionNiveau_de_gris_2.setObjectName("actionNiveau_de_gris_2")
            self.actionNiveau_de_Gris = QtWidgets.QAction(MainWindow)
            self.actionNiveau_de_Gris.setIcon(icon)
            self.actionNiveau_de_Gris.setObjectName("actionNiveau_de_Gris")
            self.actionOpen = QtWidgets.QAction(MainWindow)
            self.actionOpen.setIcon(icon1)
            self.actionOpen.setObjectName("actionOpen")
            self.actionInversion = QtWidgets.QAction(MainWindow)
            icon4 = QtGui.QIcon()
            icon4.addPixmap(QtGui.QPixmap("icons/16x16/Sorting Arrows_16px_1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionInversion.setIcon(icon4)
            self.actionInversion.setObjectName("actionInversion")
            self.actionRotation_90 = QtWidgets.QAction(MainWindow)
            self.actionRotation_90.setObjectName("actionRotation_90")
            self.actionRotation_45 = QtWidgets.QAction(MainWindow)
            self.actionRotation_45.setObjectName("actionRotation_45")
            self.actionConservation = QtWidgets.QAction(MainWindow)
            self.actionConservation.setObjectName("actionConservation")
            self.actionSymetrie = QtWidgets.QAction(MainWindow)
            self.actionSymetrie.setObjectName("actionSymetrie")
            self.actionContours = QtWidgets.QAction(MainWindow)
            self.actionContours.setObjectName("actionContours")
            self.actionTranslation_1_0_100_0_1_50 = QtWidgets.QAction(MainWindow)
            self.actionTranslation_1_0_100_0_1_50.setObjectName("actionTranslation_1_0_100_0_1_50")
            self.menuOuvrir.addAction(self.actionOpen)
            self.menuOuvrir.addAction(self.actionEnregister)
            self.menuOuvrir.addAction(self.actionExit)
            self.menuAmelioration.addAction(self.actionEgalisation)
            self.menuAmelioration.addAction(self.actionConservation)
            self.menuFiltrage.addAction(self.actionMoyenne)
            self.menuFiltrage.addAction(self.actionLissage)
            self.menuBinarisation.addAction(self.actionSeuil)
            self.menuBinarisation.addAction(self.actionErosion)
            self.menuBinarisation.addAction(self.actionDilatation)
            self.menuBinarisation.addAction(self.actionContours)
            self.menuTransformation.addAction(self.actionNiveau_de_Gris)
            self.menuTransformation.addAction(self.actionInversion)
            self.menuTransformation.addAction(self.actionSymetrie)
            self.menuRotation.addAction(self.actionRotation_90)
            self.menuTranslation.addAction(self.actionTranslation_1_0_100_0_1_50)
            self.menubar.addAction(self.menuOuvrir.menuAction())
            self.menubar.addAction(self.menuTransformation.menuAction())
            self.menubar.addAction(self.menuAmelioration.menuAction())
            self.menubar.addAction(self.menuFiltrage.menuAction())
            self.menubar.addAction(self.menuBinarisation.menuAction())
            self.menubar.addAction(self.menuRotation.menuAction())
            self.menubar.addAction(self.menuTranslation.menuAction())

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)
            self.actionOpen.triggered.connect(self.recuperer)
            self.actionEgalisation.triggered.connect(self.egalisationHistogrammeCouleur)
            self.actionNiveau_de_Gris.triggered.connect(self.grise)
            self.actionSeuil.triggered.connect(self.seuil)
            self.actionInversion.triggered.connect(self.inversion)
            self.actionErosion.triggered.connect(self.erosion)
            self.actionDilatation.triggered.connect(self.dilatation)
            self.actionLissage.triggered.connect(self.lissage)
            self.actionMoyenne.triggered.connect(self.etalage)
            self.actionEnregister.triggered.connect(self.enregister)
            self.actionSymetrie.triggered.connect(self.symetrie)
            self.horizontalSlider.valueChanged['int'].connect(self.valueLuminosite)
            self.horizontalSlider_2.valueChanged['int'].connect(self.valueBlur)
            self.actionRotation_90.triggered.connect(self.rotation)
            self.actionTranslation_1_0_100_0_1_50.triggered.connect(self.translation)

            self.filename = None #retenir adresse de l'image

            self.temp = None #affichage temporaire de l'image pour affichage

        def recuperer(self):
            self.filename = QtWidgets.QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
            self.image = cv2.imread(self.filename)
            self.affichage(self.image)

        def enregister(self):
            cv2.imwrite("Image.png",self.temp)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Enregistrement")
            msg.setInformativeText("Image enregister")
            msg.setWindowTitle("Enregister")
            msg.exec()

        def affichage(self,image):
            self.temp = image
            self.resize()
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(frame , frame.shape[1], frame.shape[0] , frame.strides[0],QImage.Format_RGB888)
            window = MainWindow.height()
            self.size1 = getLigne(window)
            self.size2 = getColonne(self.image,window)
            self.label.setMaximumSize(QtCore.QSize(self.size1, self.size2))
            self.label.setPixmap(QtGui.QPixmap.fromImage(image))

        def grise(self):
            image_grise = griser(self.filename)
            self.affichage(image_grise)

        def egalisationHistogramme(self):

            self.image = self.griser()
            equ = cv2.equalizeHist(self.image)
            self.affichage(equ)

        def egalisationHistogrammeCouleur(self):

            image = cv2.imread(self.filename)
            #on divise l'image en 3 bandes R V B
            b,v,r = cv2.split(image)
            #on egalise l'histogramme de chaque bande
            #on fusionne les 3 bandes pour avoir une seule image couleur
            y = cv2.merge((egalisationHisto(b),egalisationHisto(v),egalisationHisto(r)))
            self.affichage(y)


        def seuil(self):
            image_grise = griser(self.filename)
            seuil = otsu(image_grise)
            newimage = image_grise.copy()
            image_binaire = binarisation(image_grise,seuil,newimage)

            self.affichage(image_binaire)

        def resize(self):
            image = cv2.imread(self.filename)
            window = MainWindow.height()
            self.size1 = MainWindow.height() - 100
            self.size2 = ((MainWindow.height() - 100 ) * image.shape[0]) / image.shape[1]
            self.size1 = getLigne(window)
            self.size2 = getColonne(image,window)
            self.label.setMaximumSize(QtCore.QSize(self.size1, self.size2))

        def inversion(self):
            image = cv2.imread(self.filename)
            #on divise l'image en 3 bandes R V B
            b,v,r = cv2.split(image)

            new_imageInverser = cv2.merge((inversionCouleur(r) , inversionCouleur(v) , inversionCouleur(b)))
            self.affichage(new_imageInverser )

        def erosion(self):
            image_grise = griser(self.filename)
            image_erode = image_grise.copy()
            new_imageErode = erosion(image_grise,image_erode)
            self.affichage(new_imageErode)


        def dilatation(self):
            image_grise = griser(self.filename)
            image_dilate = image_grise.copy()
            ligne = image_grise.shape[0]
            colonne = image_grise.shape[1]
            new_imageDilate = dilatation(image_grise,ligne,colonne,image_dilate)

            self.affichage(new_imageDilate)

        def conservateur(self):
            #image_grise = griser(self.filename)
            #image_conserver = image_grise.copy()
            #new_imageConserver = conserver(image_grise,image_conserver)
            image = cv2.imread(self.filename)
            image_conserver = image.copy()
            #on divise l'image en 3 bandes R V B
            b,v,r = cv2.split(image)

            new_imageConserver = cv2.merge((conserver(b,image_conserver),conserver(v,image_conserver),conserver(r,image_conserver)))
            self.affichage(new_imageConserver)


        def lissage(self):
            image = cv2.imread(self.filename)
            image_lisse = image.copy()
            #on divise l'image en 3 bandes R V B
            b,v,r = cv2.split(image)
            image_lisseRVB = cv2.merge((lisse(v,image_lisse),lisse(b,image_lisse),lisse(r,image_lisse)))
            self.affichage(image_lisseRVB)

        def valueLuminosite(self,value):
            image = cv2.imread(self.filename) 
            self.luminosite_value_now = value
            new_imageChanger = luminosite(image,value)
            self.affichage(new_imageChanger)
            
        def valueBlur(self,value):
            image = cv2.imread(self.filename)
            self.blur_value_now = value
            blur_image =  floutter(image,value)
            self.affichage(blur_image)

        def etalage(self):
            image = cv2.imread(self.filename)
            #on divise l'image en 3 bandes R V B et on l'etale un par un puis le mettre en une seule image
            b,v,r = cv2.split(image)
            image_etale = cv2.merge((etale(b),etale(v),etale(r)))
            self.affichage(image_etale)

        def symetrie(self):
            image = cv2.imread(self.filename)
            image_sym = image.copy()
            for i in range(0,image.shape[0]):
                for j in range(0,image.shape[1]):
                  image_sym[i][j] = image[image.shape[0] - 1 - i][j]

            self.affichage(image_sym)

        def rotation(self):
            image = self.temp
            image_rot = rotation90(image)
            self.affichage(image_rot)
        
        def translation(self):
            image = cv2.imread(self.filename)
            image_translation = translation(image)
            
            self.affichage(image_translation)
            
            
            
        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            self.label_2.setText(_translate("MainWindow", "Lumonisite"))
            self.label_3.setText(_translate("MainWindow", "Floutter"))
            self.menuOuvrir.setTitle(_translate("MainWindow", "Ouvrir"))
            self.menuAmelioration.setTitle(_translate("MainWindow", "Amelioration"))
            self.menuFiltrage.setTitle(_translate("MainWindow", "Filtrage"))
            self.menuBinarisation.setTitle(_translate("MainWindow", "Binarisation"))
            self.menuTransformation.setTitle(_translate("MainWindow", "Transformation"))
            self.menuRotation.setTitle(_translate("MainWindow", "Rotation"))
            self.menuTranslation.setTitle(_translate("MainWindow", "Translation"))
            self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
            self.toolBar_2.setWindowTitle(_translate("MainWindow", "toolBar_2"))
            self.actionOuvrir.setText(_translate("MainWindow", "Ouvrir"))
            self.actionEnregister.setText(_translate("MainWindow", "Enregister"))
            self.actionExit.setText(_translate("MainWindow", "Quitter"))
            self.actionEgalisation.setText(_translate("MainWindow", "Egalisation Histogramme"))
            self.actionMedian.setText(_translate("MainWindow", "Median"))
            self.actionGaussien.setText(_translate("MainWindow", "Gaussien"))
            self.actionMoyenne.setText(_translate("MainWindow", "Etalage"))
            self.actionLissage.setText(_translate("MainWindow", "Lissage"))
            self.actionSeuil.setText(_translate("MainWindow", "Seuillage Otsu"))
            self.actionErosion.setText(_translate("MainWindow", "Erosion"))
            self.actionDilatation.setText(_translate("MainWindow", "Dilatation"))
            self.actionNiveau_de_gris.setText(_translate("MainWindow", "Niveau de gris"))
            self.actionNiveau_de_gris_2.setText(_translate("MainWindow", "Niveau de gris"))
            self.actionNiveau_de_Gris.setText(_translate("MainWindow", "Niveau de Gris"))
            self.actionOpen.setText(_translate("MainWindow", "Ouvrir"))
            self.actionInversion.setText(_translate("MainWindow", "Inversion Couleur"))
            self.actionRotation_90.setText(_translate("MainWindow", "Rotation 90 "))
            self.actionRotation_45.setText(_translate("MainWindow", "Rotation 45"))
            self.actionSymetrie.setText(_translate("MainWindow", "Symetrie"))
            self.actionContours.setText(_translate("MainWindow", "Contours"))
            self.actionTranslation_1_0_100_0_1_50.setText(_translate("MainWindow", "Translation [[1,0,100],[0,1,50]]"))
    
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())