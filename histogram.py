import numpy as np
import cv2 as cv
from math import sqrt
from numpy.core.fromnumeric import shape
from filter import Filter

class Histogram(Filter) :
    
    def __init__(self, img,freqeuncyFilteredImage ,mixedImage, imageOne,imageTwo):
        self.img = img
        self.mixedImage=mixedImage
        self.imageOne=imageOne
        self.imageTwo=imageTwo
        self.freqeuncyFilteredImage=freqeuncyFilteredImage
        self.noiseImageData=super().noiseImageData
        self.originalImageData=super().originalImageData
    def freqFilters(self,value,image=[[None]]):
        '''
        get the filter index choosen from the Frequency Filters ComboBox and apply
        that filter on the noiseImageData 
        '''
        if image[0][0]:
            original = np.fft.fft2(image)
            shape=image.shape        
        else : 
            original = np.fft.fft2(self.noiseImageData)
            shape=self.noiseImageData.shape

        center = np.fft.fftshift(original)
        if(value == 0):
            resault = center * self.idealFilterLP(50,shape)
        else:
            resault = center * self.idealFilterHP(50,shape)
        final = np.fft.ifftshift(resault)
        inverse_final = np.fft.ifft2(final)
        if not image[0][0]:
            self.freqeuncyFilteredImage.setImage(np.abs(inverse_final).T)
        else : 
            return np.abs(inverse_final).T

        
    
    def distance(self,point1,point2):
        return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def idealFilterHP(self,D0,imgShape):
        base = np.ones(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                if self.distance((y,x),center) < D0:
                    base[y,x] = 0
        return base
    def idealFilterLP(self,D0,imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                if self.distance((y,x),center) < D0:
                    base[y,x] = 1
        return base    
        
    # This part for Hybrid Images ###
    def hybrid_img(self):
        img1=cv.imread('images/image1.jpg',0)
        img2=cv.imread('images/image2.jpg',0)
        img1=cv.resize(img1,(255,255))
        img2=cv.resize(img2,(255,255))
        img1 = np.fft.fft2(img1)
        img2 = np.fft.fft2(img2)
        center1 = np.fft.fftshift(img1)
        center2 = np.fft.fftshift(img2)
        shape1=img1.shape
        shape2=img2.shape
        lowPass= center1 * self.idealFilterLP(25,shape1)
        highPass = center2 * self.idealFilterHP(5,shape2)
        finalLowPass = np.fft.ifftshift(lowPass)
        inverse_finalLowPass = np.fft.ifft2(finalLowPass)
        finalHighPass = np.fft.ifftshift(highPass)
        inverse_finalHighPass = np.fft.ifft2(finalHighPass)
        img1=np.abs(inverse_finalLowPass).T
        img2=np.abs(inverse_finalHighPass).T
        self.imageOne.setImage(img1)
        self.imageTwo.setImage(img2)
        hybrid =img1+img2
        self.mixedImage.setImage(hybrid)