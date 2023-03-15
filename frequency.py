import numpy as np
import cv2 as cv
from math import sqrt
from filter import Filter
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Frequency:
    
    def __init__(self, img, equalizedImage, normalizedImage,localThresholdImage,globalThesholdImage,grayScaleImageData,globalThreshSlider):
        self.img = img
        self.equalizedImage=equalizedImage
        self.normalizedImage=normalizedImage
        self.localThresholdImage=localThresholdImage
        self.globalThesholdImage=globalThesholdImage
        self.globalThreshSlider=globalThreshSlider
        self.grayScaleImageData=cv.cvtColor(grayScaleImageData, cv.COLOR_BGR2GRAY)
    
    def df(img):
        values = [0]*256
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                values[img[i,j]]+=1
        return values

    # @staticmethod()
    ## This part for Equalized Image ###
    def cdf(hist):
        cdf = [0] * len(hist)
        cdf[0] = hist[0]
        for i in range(1, len(hist)):
            cdf[i]= cdf[i-1]+hist[i]
        cdf = [ele*255/cdf[-1] for ele in cdf]
        return cdf

    def equalize_image(self,image):
        my_cdf = Frequency.cdf(Frequency.df(image))
        image_equalized = np.interp(image, range(0,256), my_cdf)
        return image_equalized


    ## This part for Normalized Image ###
    def normalize_image(self,img):
        minValue = min(img.flatten())
        maxValue = max(img.flatten())
        mean=np.mean(img.flatten())
        std=np.std(img.flatten())
        values = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # values[i,j] = (img[i,j] - minValue)/(maxValue - minValue) * 255.0
                values[i,j]=((img[i,j]-mean)/std**2)*1.0
        return values


    def globalThreshSliderChange(self):
        value=self.globalThreshSlider.value()    # part missing
        local_data=self.global_threshold(self.grayScaleImageData,value)
        self.globalThesholdImage.setImage(local_data.T)
    ## This part for Global Thresholding ###
    def global_threshold(self,nor_image, threshold):
        image = np.array(nor_image)
        new_img = np.copy(image)
        try:
            for channel in range(image.shape[2]):
                new_img[:, :, channel] = list(map(lambda row: list((255 if ele>threshold else 0) for ele in row) , image[:, :, channel]))
        except:
            new_img[:, :] = list(map(lambda row: list((255 if ele>threshold else 0) for ele in row) , image[:, :]))
        return new_img

   
    def localThreshSliderChange(self):
        value=self.localThreshSlider.value() # part missing
        local_data=self.local_threshold(self.grayScaleImageData,5,value)
        self.localThresholdImage.setImage(local_data.T)
    ## This part for Local Thresholding ###
    def local_threshold(self,nor_image, size, const):
        image = np.array(nor_image)
        new_img = np.copy(image)
        for row in range(0, image.shape[0], size):
            for col in range(0, image.shape[1], size):
                mask = image[row:row+size,col:col+size]
                threshold = np.mean(mask)-const
                new_img[row:row+size,col:col+size] = self.global_threshold(mask, threshold)
        return new_img

    ## This part for RGB 2 Gray_scale conversion ###
    def gry_conv(image):
        gry_img = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        return gry_img    