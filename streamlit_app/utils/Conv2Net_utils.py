# Authors:
# Hajer Souaifi-Amara
# Creation date: 21MAR2023 by Hajer
# Modification date: 21MAR2023 by Hajer

import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd

from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

from time import time

## PARAMETERS

batch_size = 32
img_height = 256
img_width  = 256
#classes = ["ART","BAS","EOS","ERY","IG","LYM","MON","MYB","NEU","PLT"]
#label_map = {'ART': 0, 'BAS': 1, 'EOS': 2, 'ERY': 3, 'IG': 4, 'LYM': 5, 'MON': 6, 'MYB': 7, 'NEU': 8, 'PLT': 9}
classes = ["Artefact","Basophil","Eosinophil","Erythrocyte","IG","Lymphocyte","Monocyte","Myeloblast","Neutrophil","Platelet"]
label_map = {'Artefact': 0, 'Basophil': 1, 'Eosinophil': 2, 'Erythrocyte': 3, 'IG': 4, 'Lymphocyte': 5, 'Monocyte': 6, 'Myeloblast': 7, 'Neutrophil': 8, 'Platelet': 9}



# Load model :
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('./data/models/Conv2Net/LenetV7_2.h5', compile = False)
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy") 
    #model.summary()
    return model


def make_heatmap(img_array, model, last_conv_layer, class_index):

    # Désactive softmax :
    model.layers[-1].activation = None
    
    t_temp = time()
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_index]
    print("makeheatmap - prediction :",time()-t_temp)
    
    t_temp = time()
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))
    print("makeheatmap - tapegradient :", time()-t_temp)
    
    heatmap_tmp = last_conv_layer_output[0].numpy()

    # Multiplie chaque carte d'activation par le gradient, puis moyenne
    for i in range(last_conv_layer_output.shape[3]):
        heatmap_tmp[:,:,i] *= pooled_grads[i]
    heatmap = np.mean(heatmap_tmp, axis=-1)
    
    # Réactive softmax :
    model.layers[-1].activation = tf.keras.activations.softmax
    
    return heatmap


def gradcam(model, img, img_orig, last_conv_layer,
            img_height, img_width, class_index, 
            alpha = 0.5):
      
    heatmap = make_heatmap(img, model, last_conv_layer, class_index)
    
    return heatmap


def print_gradcam(heatmap, img_orig, alpha = 0.8):
    """
    Traitement de la heatmap produite par make_heatmap :
        - applique ReLU
        - normalise
        - superpose la heatmap à l'image d'origine
    """

    heatmap = np.maximum(0, heatmap)
    heatmap = heatmap/heatmap.max()

    ## Superposition de l'image "img_orig" et de la heatmap
    # 1/ Rescale heatmap: 0-255
    heatmap = np.uint8(255*heatmap)
    # 2/ Jet colormap
    jet = cm.get_cmap("jet")
    # 3/ Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # 4/ Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_orig.shape[1], img_orig.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    # 6/ Superimpose the heatmap on original image
    superimposed_img = jet_heatmap*alpha + img_orig
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


# Image Preprocessing :
def get_img_array(img_file, size = (img_height, img_width), preprocess = False):
    
    img = Image.open(img_file)
    img = img.convert('RGB')
    img = img.resize(size)
    img = np.array(img)
    print("Array Dims :",img.shape)
    
    # Pour prediction 
    if preprocess == True:
        img = np.expand_dims(img, axis = 0)
        img = preprocess_input(img)
    return img

@st.cache_resource()
def preprocessing(img_file, size = (img_height, img_width)):
    img = get_img_array(img_file, size = (img_height, img_width))
    img_orig = get_img_array(img_file, size = (img_height, img_width), preprocess = False)
    return img, img_orig


# Main function
def print_proba(proba):
    if proba < 0.0001:
        return str('< 0.01%')
    
    return str(np.round(proba*100, 2))+'%'

def Conv2Net_prediction(model,img_file):
    
    # Preprocessing de l'image
    img, img_orig = preprocessing(img_file, size = (img_height, img_width))

    # Prediction :
    preds = model.predict(img)[0]
    
    sorted_indexes = np.flip(np.argsort(preds))
    sorted_preds = [preds[i] for i in sorted_indexes]
    sorted_classes = [classes[i] for i in sorted_indexes]
    
    print("Sorted Indexes :",sorted_indexes[:3])
    print("Shape :", len(sorted_indexes[:3]))
    print("Type :", type(sorted_indexes[:3]))
    
    # Détecte la dernière couche de convolution du modèle :
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = model.get_layer(layer.name)
            break     
    
    # Grad-CAM :
    fig = plt.figure(figsize = (5,5))
       
    ## Calcul de la heatmap:
    heatmap = gradcam(model, img, img_orig, last_conv_layer,
                      img_height, img_width, 
                      class_index = sorted_indexes[0], alpha = 0.8)
    ## Traitement de la heatmap:
    superimposed_img = print_gradcam(heatmap, img_orig, alpha = 0.8)
    
    ## Plot
    plt.imshow(superimposed_img)
    #plt.title(sorted_classes[0] +' (%s)'%(print_proba(sorted_preds[0])), fontsize = 14)
    plt.title('Grad-Cam', fontsize = 18)
    #plt.text(x = 10, y = 25, s = 'P(%s) = %s'%(sorted_classes[0], print_proba(sorted_preds[0])), fontsize = 'xx-large')
    #plt.text(x = 10, y = 55, s = 'P(%s) = %s'%(sorted_classes[1], print_proba(sorted_preds[1])), fontsize = 'xx-large')
    #plt.text(x = 10, y = 85, s = 'P(%s) = %s'%(sorted_classes[2], print_proba(sorted_preds[2])), fontsize = 'xx-large')
    plt.grid(None)
    plt.axis('off')

    return fig, sorted_classes, sorted_preds