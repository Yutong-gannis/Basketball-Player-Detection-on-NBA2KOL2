# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:38:40 2022

@author: lenovo
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

def transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.average(img.reshape((img.shape[0] * img.shape[1],3)), axis=0)
    return img

def detect_team(X, centroid='random'):
    cluster = KMeans(n_clusters=2, n_init=1, init=centroid).fit(X)
    centroid=cluster.cluster_centers_
    clt = cluster.labels_
    return clt, centroid

'''test
img1 = cv2.imread('test2.jpg')
img1 = transform(img1)

img2 = cv2.imread('test3.jpg')
img2 = transform(img2)

img3 = cv2.imread('test4.jpg')
img3 = transform(img3)
X = [img1, img2, img3]
'''


