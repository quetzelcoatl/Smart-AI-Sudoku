import cv2
from matplotlib import pyplot as plt

image = cv2.imread('output_grid.PNG')
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)
org = (60, 60)
fontScale = 1
thickness = 2

image = cv2.putText(image, '2', org, font,
                   fontScale, color, thickness, cv2.LINE_AA)

plt.imshow(image)
plt.show()