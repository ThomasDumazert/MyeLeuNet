###################################################################
#Author : **Hajer SOUAIFI-AMARA**

#Creation date : 04-MAR-2023

#Final date : 17-MAR-2023

#Modification date : 17-MAR-2023 - transformation en .py

##################################################################


#################################################################
#       Classification matrix and report Algo function          #
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
import tensorflow_datasets as tfds
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers


from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import itertools


##################################################################################################################
### Fonction print_classification_report: qui affiche la matrice de confusion en % et en nombre d'observations

def print_classification_report(y_true, model, labels):
    
    '''
    Args:
    - y_true: DataFrameIterator of the test dataset
    - model: keras model
    - labels: labels of the cells categories
    '''
    
    # Prédiction : utilise le jeu de test (testing_set)
    predictions = model.predict(y_true)
    y_pred = tf.argmax(predictions, axis = 1)

    # Calcul et affichage de la matrice de confusion
    cnf_matrix = confusion_matrix(y_true.classes, y_pred, normalize = 'true')
    classes = range(len(labels))
  
    plt.figure(figsize = (8,8))
    plt.imshow(cnf_matrix, interpolation = 'nearest', cmap = 'Blues')
    plt.title("Confusion matrix (with %)")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, np.around(cnf_matrix[i, j], decimals = 3),
             horizontalalignment = "center",
             color = "white" if cnf_matrix[i, j] > (cnf_matrix.max() / 2) else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()
        
    cnf_matrix = metrics.confusion_matrix(y_true.classes, y_pred)
    
    plt.figure(figsize = (8,8))

    plt.imshow(cnf_matrix, interpolation='nearest',cmap='Blues')
    plt.title("Confusion matrix (nb of obs)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > (cnf_matrix.max() / 2) else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()

    # Rapport de classification 
    report = classification_report(y_true.classes, y_pred, 
                                 target_names = labels, output_dict = True)
    
    df_report = pd.DataFrame(report).transpose()
      
    print("Classification Report")
    #print(df_report)
    return display(df_report)
