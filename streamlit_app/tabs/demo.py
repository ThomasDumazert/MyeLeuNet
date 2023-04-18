# Authors:
# Hajer Souaifi-Amara
# Creation date: 21MAR2023 by Hajer
# Modification date: 21MAR2023 by Hajer
# Modification date: 29MAR2023 by Hajer

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from utils import utils_demo
#from utils import Conv2Net_utils
from utils import FTvgg16_utils
from utils import FTmobilenetV2_utils

title = "Démonstration"
sidebar_name = "Démonstration"



#model_list = ["LeNet-5", "Fine-tuned VGG16", "Fine-tuned MobileNet V2"]
model_list = ["Fine-tuned MobileNet V2","Fine-tuned VGG16"]
def run():

    st.title(title)

    st.markdown("---")
    
    st.subheader('1. Choisissez votre modèle de classification')

    model_choice = st.selectbox("Classifieur", options=model_list)

    st.write('')
    st.subheader('2. Chargez vos images de frottis sanguin')
    st.markdown(
        '*Sélectionner une des 2 méthodes de chargement d\'images*')
    cola, colb, colc = st.columns([4, 1, 4])

    with cola:
        uploaded_img = utils_demo.upload_example()

    with colb:
        st.markdown(
            '''<h3 style='text-align: center; '> 
            <br>
            <br>
            OU
            </h3>''', unsafe_allow_html=True)

    with colc:
        dict_img, example_path = utils_demo.select_examples()

    if uploaded_img:
        img_file = uploaded_img
        img_name = img_file.name

    elif example_path:
        selected_img = dict_img[example_path]
        img_file = open(selected_img, 'rb')

        try:
            img_name = img_file.name.split('/')[-1]

        except IndexError:
            st.write('Charger une ou des images à classer')

    img = img_file.read()

    img_info = Image.open(img_file)
    file_details = f"""
    Name: {img_name}
    Type: {img_info.format}
    Size: {img_info.size}
   
    """

    st.write('')
    st.subheader('3. Résultats')

    #if model_choice == "LeNet-5":
        # Importe le modèle (en cache)
    #    model = Conv2Net_utils.load_model()
        # Prédiction + Grad-CAM
    #    fig, sorted_classes, sorted_preds = Conv2Net_utils.Conv2Net_prediction(model, img_file)
    #    p0 = Conv2Net_utils.print_proba(sorted_preds[0])
    #    p1 = Conv2Net_utils.print_proba(sorted_preds[1])
    #    p2 = Conv2Net_utils.print_proba(sorted_preds[2])
        
    if model_choice == "Fine-tuned VGG16":
        # Importe le modèle (en cache)
        #model = FTvgg16_utils.load_model()
        model = FTvgg16_utils.load_ft_vgg16('streamlit_app/data/models/FTvgg16/VGG16_model_weights_18MAR23-1_FT.h5')
        # Prédiction + Grad-CAM
        fig, sorted_classes, sorted_preds = FTvgg16_utils.FTvgg16_prediction(model, img_file)
        p0 = FTvgg16_utils.print_proba(sorted_preds[0])
        p1 = FTvgg16_utils.print_proba(sorted_preds[1])
        p2 = FTvgg16_utils.print_proba(sorted_preds[2])

    if model_choice == "Fine-tuned MobileNet V2":
        # Importe le modèle (en cache)
        model = FTmobilenetV2_utils.load_model()
        # Prédiction + Grad-CAM
        fig, sorted_classes, sorted_preds = FTmobilenetV2_utils.FTmobilenetV2_prediction(model, img_file)
        p0 = FTmobilenetV2_utils.print_proba(sorted_preds[0])
        p1 = FTmobilenetV2_utils.print_proba(sorted_preds[1])
        p2 = FTmobilenetV2_utils.print_proba(sorted_preds[2])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Cette image")
        st.write(" ")
        st.write(" ")
        st.image(img, width=400)
        st.caption(file_details)

    with col2:
        with st.container():
            st.subheader("est classée en :")

            st.text('P(%s) = %s' % (
                sorted_classes[0], p0))
            st.text('P(%s) = %s' % (
                sorted_classes[1], p1))
            st.text('P(%s) = %s' % (
                sorted_classes[2], p2))
        
    with col3:
        with st.container():
            st.subheader(sorted_classes[0] +' (%s)'%(p0))
            st.pyplot(fig)


