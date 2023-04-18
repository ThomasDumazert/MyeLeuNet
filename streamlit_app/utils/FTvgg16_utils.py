# Authors:
# Hajer Souaifi-Amara
# Creation date: 21MAR2023 by Hajer
# Modification date: 21MAR2023 by Hajer
# Modification date: 29MAR2023 by Hajer

import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications.vgg16 import VGG16

import numpy as np
import pandas as pd

from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

from time import time
import requests

#python3 quickstart.py


#from google.oauth2 import service_account
#from google.cloud import storage

# Create API client.
#credentials = service_account.Credentials.from_service_account_info(
#    st.secrets["gcp_service_account"]
#)
#client = storage.Client(credentials=credentials)


## PARAMETERS

batch_size = 32
img_height = 256
img_width  = 256
#classes = ["ART","BAS","EOS","ERY","IG","LYM","MON","MYB","NEU","PLT"]
#label_map = {'ART': 0, 'BAS': 1, 'EOS': 2, 'ERY': 3, 'IG': 4, 'LYM': 5, 'MON': 6, 'MYB': 7, 'NEU': 8, 'PLT': 9}
classes = ["Artefact","Basophil","Eosinophil","Erythrocyte","IG","Lymphocyte","Monocyte","Myeloblast","Neutrophil","Platelet"]
label_map = {'Artefact': 0, 'Basophil': 1, 'Eosinophil': 2, 'Erythrocyte': 3, 'IG': 4, 'Lymphocyte': 5, 'Monocyte': 6, 'Myeloblast': 7, 'Neutrophil': 8, 'Platelet': 9}


# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
#@st.cache_data(ttl=600)
#def read_file(bucket_name, file_path):
#    bucket = client.bucket(bucket_name)
#    content = bucket.blob(file_path).download_as_string().decode("utf-8")
#    return content

#bucket_name = "streamlit-bucket"
#file_path = "VGG16_model_18MAR23-1_FT.h5"

#content = read_file(bucket_name, file_path)

# Load model :
@st.cache_resource()
#def load_model():
#    model = tf.keras.models.load_model('streamlit_app/data/models/FTvgg16/VGG16_model_18MAR23-1_FT.hdf5') #le fichier du modèle est trop lourd pour le github, il sera hébergé sur le GoogleDrive
#    model.summary()
#    return model

@st.cache_resource()
def load_ft_vgg16(saved_weights_path):
    height = 256
    width = 256
    height_crop = 180
    width_crop = 180
    num_cat=10

    data_preprocessing = Sequential(
        [
            layers.Resizing(height=height, width=width, interpolation='nearest', crop_to_aspect_ratio=False), # redimensionne un lot d'images à une taille cible.
            layers.CenterCrop(height=height_crop, width=width_crop), #retourne une culture centrale d'un lot d'images
            layers.Rescaling(scale=1./255), # remet à l' échelle et les décalages des valeurs d'un lot d'images 
        ]
    )

    inputs = layers.Input(shape = (height, width, 3))
    x = data_preprocessing(inputs)
    base_model = VGG16(include_top = False,
                        weights = 'imagenet',
                        input_tensor = inputs,
                        input_shape = (height_crop, width_crop, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_cat, activation='softmax')(x)
    model = Model(inputs, outputs)

    for layer in base_model.layers: 
        layer.trainable = False
    base_model.trainable = False

    model.load_weights(saved_weights_path)

    return model

#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
def load_ft_vgg16(saved_weights_path):
    height = 256
    width = 256
    height_crop = 180
    width_crop = 180
    num_cat=10

    data_preprocessing = Sequential(
        [
            layers.Resizing(height=height, width=width, interpolation='nearest', crop_to_aspect_ratio=False), # redimensionne un lot d'images à une taille cible.
            layers.CenterCrop(height=height_crop, width=width_crop), #retourne une culture centrale d'un lot d'images
            layers.Rescaling(scale=1./255), # remet à l' échelle et les décalages des valeurs d'un lot d'images 
        ]
    )

    inputs = layers.Input(shape = (height, width, 3))
    x = data_preprocessing(inputs)
    base_model = VGG16(include_top = False,
                        weights = 'imagenet',
                        input_tensor = inputs,
                        input_shape = (height_crop, width_crop, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_cat, activation='softmax')(x)
    model = Model(inputs, outputs)

    for layer in base_model.layers: 
        layer.trainable = False
    base_model.trainable = False

    model.load_weights(saved_weights_path)

    return model

# @st.cache
# def load_model():

#     save_dest = Path('./data/models/FTvgg16/')
#     save_dest.mkdir(exist_ok=True)
    
#     f_checkpoint = Path("./data/models/FTvgg16/VGG16_model_18MAR23-1_FT.h5")

#     if not f_checkpoint.exists():
#         with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
#             #from GD_download import download_file_from_google_drive
#             download_file_from_google_drive(cloud_model_location, f_checkpoint)

#    model = tf.keras.models.load_model(f_checkpoint)
#    model.summary()
#    return model


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
def get_img_array(img_file, size = (img_height, img_width), preprocess = True):
    
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

def FTvgg16_prediction(model,img_file):
    
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
