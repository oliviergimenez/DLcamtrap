#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:54:26 2019

@author: gdussert
"""

import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

model_path = "snapshots_all/resnet50_csv_10.h5"
tresh=0.5

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

model = models.convert_model(model)

id_to_class=["blaireaux","chamois","chat forestier","chevreuil","lièvre","lynx","renard","sangliers","cerf"]

def show_detection(filename):
     # load image
     image = read_image_bgr(filename)

     # copy to draw on
     draw = image.copy()
     draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

     # preprocess image for network
     image = preprocess_image(image)
     image, scale = resize_image(image)

     # process image
     start = time.time()
     boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
     print("processing time: ", time.time() - start)

     # correct for image scale
     boxes /= scale

     # visualize detections
     for box, score, label in zip(boxes[0], scores[0], labels[0]):
         # scores are sorted so we can break
         if score < 0.5:
             break

         color = label_color(label)

         b = box.astype(int)
         draw_box(draw, b, color=color)

         caption = "{} {:.3f}".format(id_to_class[label], score)
         draw_caption(draw, b, caption)

     plt.figure(figsize=(15, 15))
     plt.axis('off')
     plt.imshow(draw)
     plt.show()

def show_detection_folder(folder):
     for file in os.listdir(folder):
          if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".png"):
               file_path=os.path.join(folder, file)
               print(file_path)
               show_detection(file_path)

def pred_bbox(filename):
     # load image
     image = read_image_bgr(filename)

     # copy to draw on
     draw = image.copy()
     draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

     # preprocess image for network
     image = preprocess_image(image)
     image, scale = resize_image(image)

     # process image

     boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
     return boxes, scores, labels

def comp_exif(folder):
     from iptcinfo3 import IPTCInfo
     import pandas as pd

     TP=[0]*len(id_to_class) #true positive
     FP=[0]*len(id_to_class) #false positive
     FN_void=[0]*len(id_to_class)
     FN_false=[0]*len(id_to_class)
     dict_error_FP={}
     dict_nb={}

     start = time.time()
     nb=0
     for file in os.listdir(folder):
          if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".png"):
               nb+=1
               if nb%100==0:
                    print("Done {0} images in ".format(nb), time.time() - start)

               file_path=os.path.join(folder, file)

               species=IPTCInfo(file_path)['keywords'][0].decode("utf-8")

               boxes, scores, labels = pred_bbox(file_path)

               if scores[0][0]>tresh:
                    pred_class_name=id_to_class[labels[0][0]]
                    pred_id=labels[0][0]
               else:
                    pred_class_name=None

               try:
                    dict_nb[species]+=1
               except:
                    dict_nb[species]=1

               print(file_path,species,pred_class_name)

               #Si il n'y a pas de prediction
               if pred_class_name==None:
                    if species in id_to_class:
                         species_id=id_to_class.index(species)
                         FN_void[species_id]+=1

               #S'il y a des predictions on prend celle avec la confidence al plus élevée
               else:
                    if pred_class_name==species: #bien prédit
                         TP[pred_id]+=1
                    else: #mal prédit : faux positif, si c'était une vraie classe il y a aussi un faux negatif
                         FP[pred_id]+=1
                         if species in id_to_class:
                              species_id=id_to_class.index(species)
                              FN_false[species_id]+=1
                         try:
                              dict_error_FP[species]+=1
                         except:
                              dict_error_FP[species]=1

     d = {'species': id_to_class, 'TP': TP, 'FP':FP,'FN_false':FN_false,'FN_void':FN_void}
     df = pd.DataFrame(data=d)
     print(df)

     print("\nImages source de FP par classe :")
     for k in dict_error_FP.keys():
          print(k,dict_error_FP[k])

     print("\nNombre total d'image par classe:")
     for k in dict_nb.keys():
          print(k,dict_nb[k])

     return df

comp_exif("/beegfs/data/gdussert/projects/olivier_pipeline/all_classes/test/")
