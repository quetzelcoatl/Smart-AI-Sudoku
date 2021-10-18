import tensorflow as tf
import keras_preprocessing.image
from time import process_time

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from pandas import *
#import image
import os

def preprocessing(imgpath):
    patchimage = keras_preprocessing.image.load_img(imgpath, target_size=(32, 32))
    xtemp = keras_preprocessing.image.img_to_array(patchimage)
    xtemp = xtemp.reshape(1, 32, 32, 3)
    tempimg = xtemp.astype('float32')
    tempimg = tempimg / 255.0

    return tempimg

nmodel = tf.keras.models.load_model('newmodel3/modelnumber')
#checkpointer = load_model('weights.h5')
#print(checkpointer.summary())
#ans = checkpointer.predict(patchimage)

patches_path = 'newpatches/patch_56.jpg'

#patchpath = os.path.join(patches_path, ele)
patchimage = preprocessing(patches_path)
ans = nmodel.predict(patchimage)
number = max(ans[0])
flag = 0
for key, ele in enumerate(ans[0]):
    if int(key) == 9:
        print(ele, 0)
    else:
        print(ele, int(key) + 1)
        # print(int(key))
print("^$&$^")
print(ans[0][0], 1)
print(ans[0][6], 7)