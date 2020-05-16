#!/usr/bin/env python3
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from os.path import join

model_path = "retinanet_tutorial_best_weights.h5" # model file (must be based on RESNET50)
test_dir = "test" # test directory with the target images
classfile = "class.csv" # list of class labels

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

import pandas as pd
csv = pd.read_csv(classfile,header=None)
id_to_class = list(csv[0])

def print_detection(filename): #display the detection score,label
     # load image
     image = read_image_bgr(filename)

     # preprocess image for network
     image = preprocess_image(image)
     image, scale = resize_image(image)

     # process image
     boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

     # correct for image scale
     boxes /= scale

     # display detections
     for box, score, label in zip(boxes[0], scores[0], labels[0]):
         # scores are sorted so we can break
         if score < 0.5:
             break
         print(filename, id_to_class[label], score, [int(x) for x in box])
         
     return boxes, scores, labels


for file in os.listdir(test_dir):
     if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".JPG") or file.endswith(".png"):
          print_detection(join(test_dir,file))

