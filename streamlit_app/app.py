# Authors:
# Thomas Dumazert
# Hajer Souaifi-Amara
# Creation date: 23FEB2023 by Thomas
# Modification date: 21MAR2023 by Hajer
# Modification date: 23MAR2023 by Thomas
# Modification date: 29MAR2023 by Hajer

from collections import OrderedDict

import streamlit as st
from PIL import Image

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import intro, eda, model, results, demo, conclu, gallery


st.set_page_config(
    layout="wide",
    page_title=config.TITLE,
    #page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
    page_icon="streamlit_app/assets/MyeLeuNet_icon.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (eda.sidebar_name, eda),
        (model.sidebar_name, model),
        (results.sidebar_name, results),
        (demo.sidebar_name, demo),
        (conclu.sidebar_name, conclu),
        (gallery.sidebar_name, gallery)
    ]
)

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def run():


    st.sidebar.image(add_logo(logo_path="streamlit_app/assets/MyeLeuNet_icon.png", width=100, height=100)) 
    st.sidebar.markdown("# Projet MyeLeuNet")

    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


    st.sidebar.image(
    "streamlit_app/assets/new-logo.webp",
    width=100,
    )
    st.sidebar.image(
    #"https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
    "streamlit_app/assets/Property-1Mines-l-PSL.png",
    width=200,
        )

if __name__ == "__main__":
    run()
