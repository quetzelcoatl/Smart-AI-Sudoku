import cv2
import os
import sys

import imutils
from imutils import contours
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from skimage.filters import threshold_local
from skimage.segmentation import clear_border


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


image = cv2.imread('21.PNG')

image = cv2.resize(image, (1000, 1000))
#blur = cv2.blur(image, (5, 5))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# bilateral = cv2.bilateralFilter(gray, 5, 5,5)
edges = cv2.Canny(gray, 50, 210)
kernel = np.ones((6, 6), np.uint8)
img_dilation = cv2.dilate(edges, kernel, iterations=1)
approx = "REUPLOAD"

cnts = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)

for c in sorted_contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    if len(approx) == 4:
        break

width = 1200
height = 1200

temp = approx
#one,two,three,four = temp[0],temp[3],temp[2],temp[1]
#print(one,two,three,four)

x = [int(approx[i][0][0]) for i in range(4)]
y = [int(approx[i][0][1]) for i in range(4)]

miny1 = 99999
minx1 = 0
miny2 = 99999
minx2 = 0

temp1 = 0
temp2 = 0
for ind in range(4):
    if miny1 > y[ind]:
        miny1 = y[ind]
        temp1 = ind

minx1 = x.pop(temp1)
y.pop(temp1)

for ind in range(3):
    if miny2 > y[ind]:
        miny2 = y[ind]
        temp2 = ind

minx2 = x.pop(temp2)
y.pop(temp2)

if minx1 < minx2:
    first = [minx1, miny1]
    second = [minx2, miny2]
else:
    first = [minx2, miny2]
    second = [minx1, miny1]

miny1 = 0
minx1 = 99999
miny2 = 0
minx2 = 99999
temp1 = 0
temp2 = 0
for ind in range(2):
    if minx1 > x[ind]:
        minx1 = x[ind]
        temp1 = ind

miny1 = y.pop(temp1)
x.pop(temp1)
third = [x[0],y[0]]
fourth = [minx1, miny1]

for ind in range(4):
    if miny1 > int(approx[ind][0][1]):
        miny1 = int(approx[ind][0][1])
        temp = ind

print(first,second,third,fourth)
#input = np.float32(temp)
#input = np.float32([one, two, three, four])
input = np.float32([first,second,third,fourth])
output = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

matrix = cv2.getPerspectiveTransform(input, output)

imgOutput = cv2.warpPerspective(image, matrix, (width, height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

cv2.imshow("result", imgOutput)
cv2.waitKey(0)
cv2.imwrite('NEWOUT',imgOutput)

#plt.imshow(img_dilation)
#plt.show()
#cv2.imshow('asas',img_dilation)
#cv2.waitKey()

# cv2.imshow('asas',edges)
# cv2.waitKey()
