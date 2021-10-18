import cv2
import tensorflow as tf
import keras_preprocessing.image
from time import process_time

from PIL import Image
from pandas import *
#import image
import os

def preprocessing(imgpath):
    patchimage = keras_preprocessing.image.load_img(imgpath, target_size=(32, 32))
    #patchimage = keras_preprocessing.image.load_img(imgpath, color_mode="grayscale", target_size=(28, 28))
    xtemp = keras_preprocessing.image.img_to_array(patchimage)
    xtemp = xtemp.reshape(1, 32, 32, 3)
    #xtemp = xtemp.reshape(1, 28, 28, 1)
    tempimg = xtemp.astype('float32')
    tempimg = tempimg / 255.0

    return tempimg

nmodel = tf.keras.models.load_model('newmodel3/modelnumber')

counter = -1
temp = []
ans = 0
grid = []
patches_path = 'newpatches/'
for ele in os.listdir(patches_path):
    #print(ele)
    counter += 1
    if counter == 9:
        grid.append(temp)
        temp = []
        counter = 0
    patchpath = os.path.join(patches_path, ele)
    patchimage = preprocessing(patchpath)
    ans = nmodel.predict(patchimage)
    number = max(ans[0])

    imagepatch = cv2.imread(patchpath, 0)
    imagepatch = cv2.resize(imagepatch, (32, 32))
    count = cv2.countNonZero(imagepatch)
    print(count, number, patchpath)
    if count < 30 or number < 0.5:
        temp.append(0)
        continue

    for key, ele2 in enumerate(ans[0]):
        if number == ele2:
            if int(key) == 9:
                temp.append(0)
            else:
                temp.append(int(key) + 1)

grid.append(temp)
print(DataFrame(grid))
#print(grid)
with open("grid.txt", "w") as text_file:
    for i in range(0,9):
        for j in range(0,9):
            text_file.write((str(grid[i][j])))
        text_file.write("\n")


#print(ans)