# Authors:
# Hajer Souaifi-Amara
# Creation date: 28MAR2023 by Hajer
# Modification date: 28MAR2023 by Hajer


import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import glob

title = "Echantillons d'images de cellules sanguines"
sidebar_name = "Echantillons d'images de cellules sanguines"

fname_dict = {'NEU': 'Neutrophil',
              'LYM': 'Lymphocyte',
              'MYB': 'Myeloblast',
              'MON': 'Monocyte',
              'EOS': 'Eosinophil',
              'IG': 'Immature Granulocyte',
              'ART': 'Artefact',
              'PLT': 'Platelet',
              'ERY': 'Erythroblast',
              'BAS': 'Basophil'
}

path_img = Path('streamlit_app/data/images_sample')


def select_bloodcells():

    folders_img = [f'{i.stem} - {fname_dict[i.stem]}'
                   for i in path_img.iterdir()]

    class_img = st.selectbox("Choississez un type de cellules sanguines", options=folders_img)

    class_path = path_img / class_img.split(' -')[0]
    
    dict_img = {image_path.stem: image_path
                for image_path in class_path.iterdir()}

    return dict_img

def run():

    st.title(title)

    st.markdown("---")

    dict_img = select_bloodcells()
    num_columns = 4
    
    with st.container():
        for i, (image_name, image_path) in enumerate(dict_img.items()):
            if i % num_columns == 0:
                cols = st.columns(num_columns)

            image = Image.open(image_path)
            img_name = image_path.name.split('/')[-1]
            file_details = f"""
            Name: {img_name}
            Type: {image.format}
            Size: {image.size}"""
            cols[i % num_columns].image(image, caption=file_details, width=256) #image_name
            
