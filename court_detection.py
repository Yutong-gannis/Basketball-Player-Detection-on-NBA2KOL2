# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:15:39 2022

@author: lenovo
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def court_detection(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 转HSV
    court_color = np.uint8([[[230,153,102]]])
    hsv_court_color = cv2.cvtColor(court_color, cv2.COLOR_BGR2HSV)
    hue = hsv_court_color[0][0][0]

    # define range of blue color in HSV - Again HARD CODED! :(
    lower_color = np.array([hue - 10,10,10])
    upper_color = np.array([hue + 10,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, lower_color, upper_color)

    # Show original image
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    #plt.title('Original Image') 
    #plt.show()

    # Show masked image
    #plt.imshow(mask, cmap='Greys')
    #plt.title('Mask')
    #plt.savefig('mask.jpg')
    #plt.show()
    
    #closed = cv2.erode(mask, None, iterations=3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closed, cmap='Greys')
    #plt.title('closed Image') 
    #plt.show()
    
    closed = cv2.erode(closed, None, iterations=6)
    closed = cv2.dilate(closed, None, iterations=6)
    #plt.imshow(closed, cmap='Greys')
    #plt.title('dilade Image') 
    #plt.show()
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask=~closed)
    #plt.imshow(res, cmap='Greys')
    #plt.title('res Image') 
    #plt.show()
    
    return res
    
img = cv2.imread('test5.jpg')
court_detection(img)
