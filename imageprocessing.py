import cv2
import os
import sys

import imutils
from imutils import contours
import numpy as np
from scipy.spatial import distance
from skimage.filters import threshold_local
from skimage.segmentation import clear_border

if not os.path.exists('newpatches'):
    os.makedirs('newpatches')
kernel22 = np.ones((4,4), np.uint8)
def centering(tempimg):
    cntss = cv2.findContours(tempimg, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cntss = imutils.grab_contours(cntss)

    if len(cntss) == 0:
        print("EMPTY IMAGE")

    c1 = max(cntss, key=cv2.contourArea)
    mask = np.zeros(tempimg.shape, dtype="uint8")
    cv2.drawContours(mask, [c1], -1, 255, -1)

    diff_dig = cv2.subtract(mask,tempimg)
    #diff_dig = cv2.dilate(diff_dig, kernel22, iterations=1)

    #kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #diff_dig = cv2.filter2D(diff_dig, -1, kernel_sharp)
    return diff_dig

image = cv2.imread('21out.jpg')

image = cv2.resize(image, (1200, 1200))

blur = cv2.blur(image, (5, 5))
blur = cv2.blur(blur, (6, 6))
#blur = cv2.blur(blur, (7, 7))
#blur = cv2.blur(blur, (8, 8))
#blur = cv2.blur(blur, (9, 9))

gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 75)

kernel = np.ones((6, 6), np.uint8)
img_dilation = cv2.dilate(edges, kernel, iterations=1)
img_inv = cv2.bitwise_not(img_dilation)

thresh = cv2.adaptiveThreshold(img_inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)
#thresh = cv2.adaptiveThreshold(img_inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
thresh = cv2.bitwise_not(thresh)

cv2.imshow('asas',thresh)
cv2.waitKey()

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    area = cv2.contourArea(c)
    if area < 100:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

print(thresh.shape)

img = thresh
sizeX = img.shape[1]
sizeY = img.shape[0]
nRows, mCols = 9, 9
#kernel = np.ones((5,5),np.uint8)
for i in range(0, nRows):
    for j in range(0, mCols):
        roi = img[i * sizeY // nRows:i * sizeY // nRows + sizeY // nRows,
              j * sizeX // mCols:j * sizeX // mCols + sizeX // mCols]

        mx = (0, 0, 0, 0)
        mx_area = 0
        cnts = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        #roi = cv2.resize(roi, (32, 32))
        for cont in cnts:
            x, y, w, h = cv2.boundingRect(cont)
            area = w * h
            if area > mx_area:
                mx = x, y, w, h
                mx_area = area
        x, y, w, h = mx

        roi = roi[y:y + h, x:x + w]
        roi = centering(roi)
        #roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('newpatches/patch_' + str(i) + str(j) + ".jpg", roi)
