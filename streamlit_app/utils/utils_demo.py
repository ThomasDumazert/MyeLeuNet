# Authors:
# Hajer Souaifi-Amara
# Creation date: 21MAR2023 by Hajer
# Modification date: 21MAR2023 by Hajer
# Modification date: 29MAR2023 by Hajer

from pathlib import Path
import streamlit as st

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


def select_examples():

    folders_img = [f'{i.stem} - {fname_dict[i.stem]}'
                   for i in path_img.iterdir()]

    class_img = st.selectbox('a. Choisir un type de cellule', options=folders_img)

    class_path = path_img / class_img.split(' -')[0]

    dict_img = {image_path.stem: image_path
                for image_path in class_path.iterdir()}

    example_path = st.selectbox(
        'b. Choisir une image', options=dict_img.keys())

    return dict_img, example_path


def upload_example():
    img_file = st.file_uploader(
        "Charger une image pour la classification",
        type=['jpg', 'png', 'tiff'],
    )

    return img_file
