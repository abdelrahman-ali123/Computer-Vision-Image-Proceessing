import numpy as np
import cv2 as cv
import random
from math import sqrt
from numpy.core.fromnumeric import shape
import cv2
from PIL import Image
import matplotlib
import math
from scipy import ndimage

class Filter:
    
    def __init__(self, originalImageData, noiseImage, edgeDetectionImage,filteredImage,noiseSliderValue,smoothingFilters,edgeDetectionFilters,img_final):
        self.originalImageData = originalImageData
        
        self.filteredImage=noiseImage
        self.noiseImage=edgeDetectionImage
        self.edgeDetectionImage=filteredImage
        self.noiseSliderValue=noiseSliderValue
        self.grayScaleImageData=cv.cvtColor(self.originalImageData, cv.COLOR_BGR2GRAY)
        self.edgeDetectionFilters=edgeDetectionFilters
        self.smoothingFilters=smoothingFilters
        self.img_final=img_final

    #add the noise and display
    def applyNoise(self,value):
        self.noiseImageData=np.array(self.grayScaleImageData.copy())
        if (value == "Guassian"):
            self.noiseImageData=self.noiseImageData+np.random.normal(0,self.noiseSliderValue**.5,self.noiseImageData.shape)
        elif(value == "Salt & Pepper"):
            prop=self.noiseSliderValue/200.0
            thresh=1-prop
            for i in range(self.grayScaleImageData.shape[0]):
                for j in range(self.grayScaleImageData.shape[1]):
                    rand=random.random()
                    if rand<prop :
                        self.noiseImageData[i][j]=0
                    elif rand>thresh:
                        self.noiseImageData[i][j]=255
        elif(value == "Uniform"):
            self.noiseImageData =self.noiseImageData+self.noiseSliderValue
        self.noiseImage.setImage(self.noiseImageData.T)
        self.noiseImage.show()
        
    # avg filters
    def avgFilter(self,value):
        '''
        get the filter index choosen from the Filtered Image ComboBox and apply
        that filter on the noiseImageData 
        '''
        avgPic = self.noiseImageData.copy()
        picShape = self.noiseImageData.shape
        filterShape = self.smoothingFilters[value].shape

        inputPicRow = picShape[0] + filterShape[0] - 1
        inputPicColumn = picShape[1] + filterShape[1] - 1
        zeros = np.zeros((inputPicRow,inputPicColumn))

        for i in range(picShape[0]):
            for j in range(picShape[1]):
                zeros[i+np.int((filterShape[0]-1)/2),j+np.int((filterShape[1]-1)/2)] = avgPic[i,j]
        if(value == 2):
            for i in range(picShape[0]):
                for j in range(picShape[1]):
                    targetWindow = zeros[i:i+filterShape[0],j:j+filterShape[1]]
                    result = np.median(targetWindow)
                    avgPic[i,j] = result
        else:
            for i in range(picShape[0]):
                for j in range(picShape[1]):
                    targetWindow = zeros[i:i+filterShape[0],j:j+filterShape[1]]
                    result = np.sum(targetWindow*self.smoothingFilters[value])
                    avgPic[i,j] = result
        self.filteredImage.setImage(avgPic.T)
    

    def edgFilters(self,value):
        '''
        get the filter index choosen from the Image Edges ComboBox and apply
        that filter on the noiseImageData 
        '''
        pic = self.noiseImageData.copy()
        maskVertical =  self.edgeDetectionFilters[value]
        maskHorizontal = maskVertical.T
        picShape = self.noiseImageData.shape
        filterShape = maskVertical.shape


        inputPicRow = picShape[0] + filterShape[0] - 1
        inputPicColumn = picShape[1] + filterShape[1] - 1
        zeros = np.zeros((inputPicRow,inputPicColumn))

        for i in range(picShape[0]):
            for j in range(picShape[1]):
                zeros[i+np.int((filterShape[0]-1)/2),j+np.int((filterShape[1]-1)/2)] = pic[i,j]

        for i in range(picShape[0]):
            for j in range(picShape[1]):
                targetWindow = zeros[i:i+filterShape[0],j:j+filterShape[1]]

                verticalResult = (targetWindow*maskVertical)
                verticalScore = verticalResult.sum() / 4

                horizontalResult = (targetWindow*maskHorizontal)
                horizontalScore = horizontalResult.sum() / 4

                result = (verticalScore**2 + horizontalScore**2)**.5
                pic[i,j] = result*3
        self.edgeDetectionImage.setImage(pic.T)
        

    def gaussian_kernel(self,size, sigma=1, verbose=False):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.dnorm(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
        kernel_2D *= 1.0 / kernel_2D.max()
        return kernel_2D

    def dnorm(x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

    def convolution(image, kernel, average=False, verbose=False):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape
        output = np.zeros(image.shape)
        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)
        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]
        return output



    #Step2: Gradient Calculation
    def sobel_filters(img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)
        
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        
        return (G, theta)


    #Step3: Non-maximum suppression
    def non_max_suppression(img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        
        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255
                    
                #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0

                except IndexError as e:
                    pass
        
        return Z

    #Step4: Double threshold
    def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        
        highThreshold = img.max() * highThresholdRatio;
        lowThreshold = highThreshold * lowThresholdRatio;
        
        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)
        
        weak = np.int32(25)
        strong = np.int32(255)
        
        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)
        
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        return (res, weak, strong)

    #Step5: Edge Tracking by Hysteresis
    def hysteresis(img, weak, strong=255):
        M, N = img.shape  
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img

    def canny_apply(self,img_name):

        kernel_size = 7
        verbose = True
        img = matplotlib.image.imread(img_name)
        kernel = self.gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
        noisy_img  = self.convolution(img, kernel, True, verbose=verbose)
        (grad_img, theta) = self.sobel_filters(noisy_img)
        n_sup_img = self.non_max_suppression(grad_img, theta)
        (res_img, weak_mat, strong_mat) = self.threshold(n_sup_img)
        self.hys_img = self.hysteresis(res_img, weak_mat, strong_mat)
        self.img_final =Image.fromarray(self.hys_img)
        return self.img_final
