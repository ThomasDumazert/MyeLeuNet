# Authors:
# Hajer Souaifi-Amara
# Creation date: 21MAR2023 by Hajer
# Modification date: 21MAR2023 by Hajer
# Modification date: 29MAR2023 by Hajer

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

title = "Conclusions & Perspectives"
sidebar_name = "Conclusions & Perspectives"


def run():

    st.title(title)

    st.markdown("---")
    
    st.markdown(
        """
        ## Conclusions

        Après de nombreux tests pour déterminer les étapes de pré-traitements les plus pertinents, nous pouvons proposer 
        3 CNN dont les performances sont très satisfaisantes pour notre problématique de classification de **10 types de cellules du sang périphérique**:

        - **LeNet-5** (accuracy = **91.6%**)
        - **VGG16 optimisé par fine-tuning** (accuracy = **92.86%**)
        - **MobileNetV2 optimisé par fine-tuning** (accuracy = **93.08%**)

        Ces performances reflètent également la taille importante (> 67 000 images) et la diversité de notre jeu de données provenant de 4 sources 
        différentes et comprenant des images variées en terme de qualité.

        Cependant, pour totalement répondre au besoin de détection de cellules sanguines responsables de l'AML, les modèles devront être améliorés afin de 
        distinguer les sous-types de progéniteurs qui constituent les granulocytes immatures (myélocytes, métamyélocytes et promyélocytes).

        Pour cela, il serait indispensable de fournir aux modèles un nombre plus important d'images de ces sous-types, ce qui semble être une tâche difficile notamment à cause de : 
        1) la grande proximité morphologique des lignages 
        2) la rareté des catégories cellulaires
        3) la difficulté de labellisation de certains sous-lignages, y compris pour des professionnels de santé. 

        """
    )

    st.markdown(
    """
    ## Perspectives

    Pour ce projet, nous avons utilisé des images qui ont toute nécessité une identification manuelle préalable de la cellule à labelliser sur une lamelle qui en contient des milliers.
    Il serait donc intéressant d'entraîner des modèles de classification, non plus à partir d'images de cellules "uniques" mais à partir d'images de lamelles entières. 
    Le but serait de détecter les cellules non-érythrocytaires dans une lamelle entière en utilisant une approche de détection d'objets combinant un RPN (Region Proposal Network) et un CNN comme le feraient 
    **Retina-Net**, **Faster R-CNN**, **Mask R-CNN**, **MobileNet-V2 SSD** ou **YOLO**.

    A terme, ce type de classifieur pourrait être embarqué dans une **application mobile** qui serait utilisée par les professionnels de santé pour accélérer leurs diagnostics.  
   
    """
    )
    st.write("[source](https://towardsdatascience.com/detection-and-classification-of-blood-cells-with-deep-learning-part-2-training-and-evaluation-53381dbbc565)")
    st.image(
    Image.open("streamlit_app/assets/yoloexample.png"), width = 800, output_format='PNG',
    caption = "Exemple de sortie d'un classifieur utilisant un algorithme de détection d'objets (credits: https://towardsdatascience.com/detection-and-classification-of-blood-cells-with-deep-learning-part-2-training-and-evaluation-53381dbbc565)"
    )
     