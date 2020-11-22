import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import get_session
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import os.path
from os import path
#from models import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import dcnn_resnet
from models import resnet50v2
from keras.preprocessing import image
tf.compat.v1.disable_v2_behavior()

import shap

import tensorflow.keras.backend as K
import json
#from src.data.preprocess import remove_text
import cv2

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def remove_text(img):
    '''
    Attempts to remove bright textual artifacts from X-ray images. For example, many images indicate the right side of
    the body with a white 'R'. Works only for very bright text.
    :param img: Numpy array of image
    :return: Array of image with (ideally) any characters removed and inpainted
    '''
    mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1][:, :, 0].astype(np.uint8)
    img = img.astype(np.uint8)
    result = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS).astype(np.float32)
    return result


img_shape = tuple(cfg['DATA']['IMG_DIM'])
y_col = 'label_str'
class_mode = 'categorical'

n_classes = 2  #len(cfg['DATA']['CLASSES'])
#histogram = np.bincount(data['TRAIN']['label'].astype(int))
output_bias = [0.0, 0.0] #np.log([histogram[i] / (np.sum(histogram) - histogram[i]) for i in range(histogram.shape[0])])
num_gpus = cfg['TRAIN']['NUM_GPUS']

thresholds = 1.0 / len(cfg['DATA']['CLASSES'])      # Binary classification threshold for a class

metrics = None
input_shape = cfg['DATA']['IMG_DIM'] + [3]
model_def = resnet50v2
#print("input_shape",input_shape)
model = model_def(cfg['NN']['DCNN_BINARY'], input_shape, metrics, 2, output_bias=output_bias, gpus=num_gpus)

# load pre-trained model 
model.load_weights('model20201115-050425.h5')

#Load Image
img_width, img_height = 224, 224
img = image.load_img(r'C:\Temp\shap2\000001-10.jpg', target_size=(img_width, img_height))

to_explain  = image.img_to_array(img).reshape(1, img_width, img_height, 3)
X = to_explain.copy().reshape(1, img_width, img_height, 3)

#Define Classes
class_names = { '0': ['negative', 'no-covid'], '1': ['positive', 'covid']  }

# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return tf.compat.v1.keras.backend.get_session().run(model.layers[layer].input, feed_dict)

e = shap.GradientExplainer((model.layers[7].input, model.layers[-1].output), map2layer(preprocess_input(X.copy()), 7))
shap_values,indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, to_explain, index_names)