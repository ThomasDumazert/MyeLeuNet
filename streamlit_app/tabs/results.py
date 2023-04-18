# Authors:
# Thomas Dumazert
# Hajer Souaifi-Amara
# Creation date: 29MAR2023 by Hajer 

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Résultats"
sidebar_name = "Résultats"

def run():

    st.title(title)

    st.markdown("---")

    col1, col2 = st.columns([1, 3])

    with col1:
     
        
        results=st.radio(
        " ",
        options=["Model summary", "Training & validation loss/accuracy", "Confusion matrix", "Performances", "Interpretability"]
        )   

    with col2:
        cont_2 = st.container()
        with cont_2.expander('Modèle Baseline: Architecture LeNet-5'):


            if results == "Model summary":
                left_co, cent_co, last_co = st.columns(3)
                with cent_co:
                    st.image(
                    Image.open("streamlit_app/data/models/Conv2Net/Schema_LeNet.png"), width = 500,
                    )
                    
            if results == "Training & validation loss/accuracy":
                left_co, cent_co,last_co = st.columns(3)
                with cent_co:
                    st.image(
                    Image.open("streamlit_app/data/models/Conv2Net/Learning_curve_Lenet.png"),
                    )
            if results == "Confusion matrix":
                left_co, last_co = st.columns(2)
                with left_co:
                    st.image(
                    Image.open("streamlit_app/data/models/Conv2Net/Conf_matrix_Lenet.png"),
                    )
            if results == "Performances":
                left_co, cent_co,last_co = st.columns(3)
                with cent_co:
                    st.image(
                    Image.open("streamlit_app/data/models/Conv2Net/Classif_report_Lenet.png"), width = 400,
                    )
            if results == "Interpretability":
                st.image(
                Image.open("streamlit_app/data/models/Conv2Net/LeNet_gradcam_goodpred_FAK.png"),
                )
                st.image(
                Image.open("streamlit_app/data/models/Conv2Net/LeNet_gradcam_badpred_FAK.png"),
                )        

        cont_4 = st.container()
        #cont_4.markdown(
        #    """
        #    ## Transfer Learning + Fine-Tuning
        #    """
        #)

        with cont_4.expander('Transfer Learning + Fine-Tuning'):
            

            FTvgg16, FTmobilenetV2 = st.tabs(['Fine-tuned VGG16', 'Fine-Tuned MobileNet V2'])

            with FTvgg16:
            
                if results == "Model summary":

                    st.image(
                    Image.open("streamlit_app/data/models/FTvgg16/FTvgg16_graph.png"), width = 800, output_format='PNG',
                    )
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(
                        Image.open("streamlit_app/data/models/Fully-Connected-Layers.png"), width = 300, output_format='PNG',
                        )
                    with last_co:
                        st.image(
                        Image.open("streamlit_app/data/models/FTvgg16/FTvgg16_params.png"), width = 250, output_format='PNG',
                        )
                        
                if results == "Training & validation loss/accuracy":
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(
                        Image.open("streamlit_app/data/models/FTvgg16/FTvgg16_loss_accuracy.png"),
                        )
                if results == "Confusion matrix":
                    left_co, last_co = st.columns(2)
                    with left_co:
                        st.image(
                        Image.open("streamlit_app/data/models/FTvgg16/FTvgg16_conf_matrix_pct.png"),
                        )
                    with last_co:   
                        st.image(
                        Image.open("streamlit_app/data/models/FTvgg16/FTvgg16_conf_matrix_nb.png"),
                        )
                if results == "Performances":
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(
                        Image.open("streamlit_app/data/models/FTvgg16/FTvgg16_performances.png"), width = 400,
                        )
                if results == "Interpretability":
                    st.image(
                    Image.open("streamlit_app/data/models/FTvgg16/FTvgg16_gradcam_goodpred.png"),
                    )
                    st.image(
                    Image.open("streamlit_app/data/models/FTvgg16/FTvgg16_gradcam_badpred.png"),
                    )


            with FTmobilenetV2:

                if results == "Model summary":

                    st.image(
                        Image.open("streamlit_app/data/models/FTmobilenet/MobileNetV2_graph.png"), width = 500, output_format='PNG',
                    )
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(
                        Image.open("streamlit_app/data/models/Fully-Connected-Layers.png"), width = 300,
                        )
                    with last_co:
                        st.image(
                        Image.open("streamlit_app/data/models/FTmobilenet/FTmobilenet_params.png"), width = 250,
                        )
                if results == "Training & validation loss/accuracy":
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(
                        Image.open("streamlit_app/data/models/FTmobilenet/FTmobilenetV2_loss_accuracy.png"),
                        )
                if results == "Confusion matrix":
                    left_co, last_co = st.columns(2)
                    with left_co:
                        st.image(
                        Image.open("streamlit_app/data/models/FTmobilenet/FTmobilenetV2_conf_matrix_pct.png"),
                        )
                    with last_co:   
                        st.image(
                        Image.open("streamlit_app/data/models/FTmobilenet/FTmobilenetV2_conf_matrix_nb.png"),
                        )
                if results == "Performances":
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(
                        Image.open("streamlit_app/data/models/FTmobilenet/FTmobilenetV2_performances.png"), width = 400,
                        )
                if results == "Interpretability":
                    st.image(
                    Image.open("streamlit_app/data/models/FTmobilenet/FTmobilenetV2_gradcam_goodpred.png"),
                    )
                    st.image(
                    Image.open("streamlit_app/data/models/FTmobilenet/FTmobilenetV2_gradcam_badpred.png"),
                    )