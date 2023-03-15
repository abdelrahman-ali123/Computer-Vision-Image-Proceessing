from numpy.core.fromnumeric import shape
import pyqtgraph as pg
from PyQt5 import QtWidgets
import cv2 as cv
from math import sqrt
import numpy as np
from UI import Ui_MainWindow
from filter import Filter
from frequency import Frequency
from histogram import Histogram
import sys
# from collections import Counter # Replaced


class GUI(Ui_MainWindow,Frequency,Histogram):
    def __init__(self,MainWindow):
        super(GUI,self).setupUi(MainWindow) 
        self.images=[self.filteredImage,self.noiseImage,self.edgeDetectionImage,
                    self.freqeuncyFilteredImage,self.equalizedImage,self.normalizedImage,
                    self.redChannel,self.greenChannel,self.blueChannel,
                    self.imageOne,self.imageTwo,self.mixedImage,self.grayScaleImage,self.globalThesholdImage,
                    self.localThresholdImage]
         
        self.smoothingFilters = [np.array([(1,1,1),(1,1,1),(1,1,1)]) * (1/9),
                        np.array([(1,2,1),(2,4,2),(1,2,1)]) * (1/16),
                        np.array([(0,0,0),(0,0,0),(0,0,0)]) ]
        self.edgeDetectionFilters = [np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]),
                                    np.array([[1, 0],[ 0, -1]]),
                                    np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]]),
                                    np.array([[-1,-1,-1],[0,0,0],[1,1,1]])]
        
        
        
        #removing unwanted options from the image display widget
        for i in range(len(self.images)):
            self.images[i].ui.histogram.hide()
            self.images[i].ui.roiPlot.hide()
            self.images[i].ui.roiBtn.hide()
            self.images[i].ui.menuBtn.hide()
            self.images[i].view.setContentsMargins(0,0,0,0)
            self.images[i].view.setAspectLocked(False)
            self.images[i].view.setRange(xRange=[0,100],yRange=[0,100], padding=0)
            
        #noise slide configrations
        self.noiseSlider.setValue(20)
        self.noiseSlider.setMaximum(50)
        self.noiseSlider.setMinimum(0) 
        self.noiseSlider.valueChanged.connect(self.noiseSliderChange)
        self.noiseSliderValue=20
        #threshold sliders
        self.localThreshValue=20
        self.globalThreshValue=125
        self.localThreshSlider.setMaximum(50)
        self.localThreshSlider.setMinimum(0)
        self.localThreshSlider.setSingleStep(2)
        self.localThreshSlider.setValue(self.localThreshValue)
        self.localThreshSlider.valueChanged.connect(super().localThreshSliderChange)
        self.globalThreshSlider.setMaximum(255)
        self.globalThreshSlider.setMinimum(0)
        self.globalThreshSlider.setSingleStep(2)
        self.globalThreshSlider.setValue(self.globalThreshValue)
        self.globalThreshSlider.valueChanged.connect(super().globalThreshSliderChange)
        #retrieve the original image datas
        self.originalImageData=cv.imread('images/test.jpg')
        #display the grayscale image
        self.grayScaleImageData=cv.cvtColor(self.originalImageData, cv.COLOR_BGR2GRAY)
        self.grayScaleImage.setImage(Frequency.gry_conv(self.originalImageData).T,scale=[2,2])
        self.grayScaleImage.show()
        #test RGB
        self.redChannel.setImage(self.originalImageData[:,:,2])
        self.redChannel.setColorMap(pg.ColorMap([0.0,1.0],[(0,0,0),(255,0,0)]))
        self.redChannel.ui.histogram.show()
        self.greenChannel.setImage(self.originalImageData[:,:,1])
        self.greenChannel.setColorMap(pg.ColorMap([0.0,1.0],[(0,0,0),(0,255,0)]))
        self.greenChannel.ui.histogram.show()
        self.blueChannel.setImage(self.originalImageData[:,:,0])
        self.blueChannel.setColorMap(pg.ColorMap([0.0,1.0],[(0,0,0),(0,0,255)]))
        self.blueChannel.ui.histogram.show()
        #link events with functions 
        self.noiseOptions.currentTextChanged.connect(super().applyNoise)
        Filter.applyNoise(self,"Uniform")

        # filters
        self.filtersOptions.currentIndexChanged.connect(super().avgFilter)
        self.edgeDetectionOptions.currentIndexChanged.connect(super().edgFilters)
        self.frequancyFiltersOptions.currentIndexChanged.connect(super().freqFilters)
        #equalization
        eq = Frequency.equalize_image(self,self.grayScaleImageData)
        self.equalizedImage.ui.histogram.show()
        self.equalizedImage.setImage(eq.T) # P.S. PlotItem type is: ImageView
        ## This part for Histogram Graph ###
        x = np.linspace(0, 255, num=256)
        y = Frequency.df(self.grayScaleImageData)
        bg = pg.BarGraphItem(x=x, height=y, width=1, brush='r')
        self.originalHistogram.addItem(bg) # P.S. PlotItem type is: PlotWidget
        #normalize
        nr = Frequency.normalize_image(self,self.grayScaleImageData)
        self.normalizedImage.ui.histogram.show()
        self.normalizedImage.setImage(nr.T) # P.S. PlotItem type is: ImageView
        #display filters
        super().avgFilter(0)
        super().edgFilters(0)
        super().freqFilters(0)
        #display hybrid image
        super().hybrid_img()
        #threshold display
        global_data=self.global_threshold(self.grayScaleImageData,self.globalThreshValue)
        self.globalThesholdImage.setImage(global_data.T)
        local_data=self.local_threshold(self.grayScaleImageData,5,self.localThreshValue)
        self.localThresholdImage.setImage(local_data.T)
    #add noise functions
    #rerender when the slider changed
    def noiseSliderChange(self):
        self.noiseSliderValue=self.noiseSlider.value()
        self.applyNoise(self.noiseOptions.currentText())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = GUI(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())