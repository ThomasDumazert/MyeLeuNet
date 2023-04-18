###################################################################
#Author : **Hajer SOUAIFI-AMARA**

#Creation date : 04-MAR-2023

#Final date : 17-MAR-2023

#Modification date : 17-MAR-2023 - transformation en .py

##################################################################


#################################################################
#                      GradCAM Algo function                    #
#################################################################

import argparse
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import sys, os
from pathlib import Path
import glob
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf

from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import itertools

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions

## SOURCE des fonctions suivantes:  https://keras.io/examples/vision/grad_cam/
# - get_img_array
# - make_gradcam_heatmap
# - save_and_display_gradcam

def get_img_array_2(img_path, TF, height, width):
    """
    Importe une image et applique le preprocessing nécessaire à l'utilisation de modèle : batch + preprocess_input
    """
    img_array = tf.keras.preprocessing.image.load_img(img_path, target_size = (height, width))
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis = 0)
    

    #data_preprocessing = tf.keras.Sequential(
    #    [
    #    layers.Resizing(height=height, width=width, interpolation='nearest',crop_to_aspect_ratio=False), # redimensionne un lot d'images à une taille cible.
    #    layers.CenterCrop(height=height_crop, width = width_crop), #retourne une culture centrale d'un lot d'images
    #    layers.Rescaling(scale=1./255), # remet à l' échelle et les décalages des valeurs d'un lot d'images 
    #    ]
    #)
    
    #img_array = data_preprocessing(img_array)
    
    if TF == 'vgg16':
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        height = width = 224
        img_array = preprocess_input(img_array) 
    elif TF == 'mobilenet':
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        height = width = 224
        img_array = preprocess_input(img_array) 
    elif TF == 'xception':
        preprocess_input = tf.keras.applications.xception.preprocess_input
        height = width = 299
        img_array = preprocess_input(img_array) 
    
    #else:
    #    img_array = data_preprocessing(img_array)
        
    return img_array



def get_img_array(img_path, height, width):
    """
    Importe une image et applique le preprocessing nécessaire à l'utilisation de modèle : batch + preprocess_input
    """
    img_array = tf.keras.preprocessing.image.load_img(img_path, target_size = (height, width))
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis = 0)
    
    img_array = preprocess_input(img_array)

    return img_array

def make_heatmap(img_array, model, last_conv_layer, class_index):
    """
    Calcule la CAM (Class Activation Map) correspondant au label d'indice "class_index" pour l'image "img_array"
    model : un modèle déjà entraîné, dont on désactive la fonction d'activation en sortie (fait dans grandcam())
    last_conv_layer : dernière couche de convolution du modèle. Trouvée automatiquement par la fonction gradcam()
    class_index : renseigné dans l'appel de gradcam() ou trouvé automatiquement par gradcam()
    """
    
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
  
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))

    heatmap_tmp = last_conv_layer_output[0].numpy()

    # Multiplie chaque carte d'activation par le gradient, puis moyenne
    for i in range(last_conv_layer_output.shape[3]):
        heatmap_tmp[:,:,i] *= pooled_grads[i]
    heatmap = np.mean(heatmap_tmp, axis=-1)

    return heatmap


def gradcam(model, TF, img_path, height, width, class_index = None, alpha = 0.5, plot = True):

    # Chargement + preprocessing de l'image:
    #img_array = get_img_array(img_path, height, width)
    img_array = get_img_array_2(img_path, TF, height, width)

    # Désactive softmax sur la dernière couche :
    model.layers[-1].activation = None

    # Détecte la dernière couche de convolution du modèle :
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = model.get_layer(layer.name)
            break        

    
    if class_index == None :
        # Trouve la classe la plus probable :
        predict = model.predict(img_array)
        class_index = np.argmax(predict[0])

    # Calcul de la CAM : resize pour superposition avec l'image finale
    heatmap = make_heatmap(img_array, model, last_conv_layer, class_index)
    big_heatmap = heatmap

      # Réactive softmax :
    model.layers[-1].activation = tf.keras.activations.softmax

      ## Traitement de la Heatmap
      # Applique ReLu (élimine les valeurs négatives de la heatmap)
    big_heatmap = np.maximum(0, big_heatmap)
      # Normalisation
    big_heatmap = big_heatmap/big_heatmap.max()

      ## Superposition de l'image et de la heatmap 
      # 1/ Import de l'image d'origine
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

      # 2/ Rescale heatmap: 0-255
    big_heatmap = np.uint8(255*big_heatmap)
      # 3/ Jet colormap
    jet = cm.get_cmap("jet")
      # 4/ Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[big_heatmap]
      # 5/ Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
      # 6/ Superimpose the heatmap on original image
    superimposed_img = jet_heatmap*alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    if plot == True:
    # 7/ Affichage des résultats
        fig = plt.figure(figsize = (8,8))
        fig.add_subplot(1,2,1)
        plt.imshow(big_heatmap)

        fig.add_subplot(1,2,2)
        plt.imshow(superimposed_img)
        plt.title("Chosen class : "+str(list(label_map.keys())[class_index]))

    return big_heatmap, superimposed_img


##################################################################################################################
### Création de la fonction grad_cam_viz: qui superpose le grad-cam aux images préprocessées pour les 10 catégories de cellules
def grad_cam_viz(df, model, TF, numcat, height, width, height_crop, width_crop, label_map):
    
    '''
    Args:
    - df: data frame with predictions
    - model: H5 or HdF5 model or base_model in case of Transfer Learning
    - TF = None or the name of the Transfer Learning model ('vgg16', 'mobilenet', 'xception')
    - numcat: number of categories
    - height: image height
    - width: image width
    - height_crop: image height after cropping
    - width_crop: image height after cropping
    - label_map: labels of the cells categories
    '''
    num_cat = numcat
    #img_size = (height, width)
    TF= TF
    
    #data_preprocessing = tf.keras.Sequential(
    #    [
    #    layers.Resizing(height=height, width=width, interpolation='nearest',crop_to_aspect_ratio=False), # redimensionne un lot d'images à une taille cible.
    #    layers.CenterCrop(height=height_crop, width = width_crop), #retourne une culture centrale d'un lot d'images
    #    layers.Rescaling(scale=1./255), # remet à l' échelle et les décalages des valeurs d'un lot d'images 
    #    ]
    #)

    # pour les images bien classées
    fig = plt.figure(figsize = (30, 20))
    plt.suptitle("Grad-Cam for Well-classified PBC",fontsize=50)
    i = 0
   
    for blood_cell in range(num_cat):
        df_temp =  df[(df["label"] == blood_cell) & (df["good_class"] == True)]
        if len(df_temp.index) == 0:
            continue
        id = np.random.choice(df_temp.index, size = 1, replace = False)
        img_path = df_temp.loc[id[0],"img_path"]
  
        #img_array = get_img_array(img_path, height, width)
        
        # Application du préprocessing prédéfinie pour chaque image
        #img_array = data_preprocessing(img_array)
        
        #heatmap1 = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        #heatmap, superimposed_img = save_and_display_gradcam(img_path, heatmap1)

        big_heatmap, superimposed_img = gradcam(model, TF, img_path, height, width,
                                          class_index = None, alpha = 0.8, plot = False)
        
        fig.add_subplot(4,6,i+1)
        plt.imshow(plt.imread(img_path))
        plt.title("Original " + list(label_map.keys())[blood_cell], fontsize = 30)
        plt.grid(None)
        plt.axis('off')

        lp = df_temp.loc[id[0],"label_pred"]
        fig.add_subplot(4,6,i+2)
        plt.imshow(superimposed_img) 
        plt.title("Grad-CAM " + list(label_map.keys())[lp], fontsize = 30) 
        plt.grid(None)
        plt.axis('off')

        i += 2
        
        
    # pour les images mal classées
    fig = plt.figure(figsize = (30, 20))
    plt.suptitle("Grad-Cam for Misclassified PBC",fontsize=50)
    i = 0
   
    for blood_cell in range(num_cat):
        df_temp =  df[(df["label"] == blood_cell) & (df["good_class"] == False)]
        if len(df_temp.index) == 0:
            continue
        id = np.random.choice(df_temp.index, size = 1, replace = False)
        img_path = df_temp.loc[id[0],"img_path"]
  
        #img_array = get_img_array(img_path, size=img_size)
        
        # Application du préprocessing prédéfinie pour chaque image
        #img_array = data_preprocessing(img_array)
        
        #heatmap1 = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        #heatmap, superimposed_img = save_and_display_gradcam(img_path, heatmap1)

        big_heatmap, superimposed_img = gradcam(model, TF, img_path, height, width,
                                          class_index = None, alpha = 0.8, plot = False)
        
        fig.add_subplot(4,6,i+1)
        plt.imshow(plt.imread(img_path))
        plt.title("Original " + list(label_map.keys())[blood_cell], fontsize = 30)
        plt.grid(None)
        plt.axis('off')

        lp = df_temp.loc[id[0],"label_pred"]
        fig.add_subplot(4,6,i+2)
        plt.imshow(superimposed_img) 
        plt.title("Grad-CAM " + list(label_map.keys())[lp], fontsize = 30) 
        plt.grid(None)
        plt.axis('off')

        i += 2
    return

