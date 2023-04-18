# Authors:
# Thomas Dumazert
# Hajer Souaifi-Amara
# Creation date: 23FEB2023 by Thomas
# Modification date: 23MAR2023 by Thomas

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px

title = "Projet MyeLeuNet"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
      """
      ## Contexte et objectif

      La **Leucémie myéloïde aigüe** (LMA) est le type de cancer du sang le plus fréquent chez l'adulte, bien qu'il soit observé à tout âge. 
      Encore très difficile à traiter, ce cancer se caractérise par le fait que des cellules souches de la moëlle osseuse ne mûrissent plus 
      pour devenir des globules blancs, mais restent immatures et remplacent rapidement les cellules sanguines normales produites par la moëlle osseuse. 

      Dans le cas de la LMA, la précocité du diagnostic est importante pour le traitement et le pronostic vital du patient.
      En plus d'un examen clinique, de tests de diagnostic, les **analyses sanguines** du patient sont une étape cruciale du dépistage car 
      elles vont permettre d'**identifier des signes de la LMA**, tels qu'une numération globulaire complète (NGC) anormale, des anomalies des cellules sanguines, 
      une baisse du nombre de plaquettes et une augmentation des globules blancs. 
      
      Les analyses des différents types de cellules du sang périphérique se font sous microscope. 
      La méthode des frottis sanguins est souvent utilisée pour colorer le sang et faciliter l'analyse. Cependant, cette technique est **chronophage** et sujette à des **biais**, 
      ce qui rend souhaitable une méthode de traitement automatisée et fiable pour un diagnostic plus rapide.

      Au travers de notre projet **MyeLeuNet**, nous allons mettre en application des techniques de **Computer Vision** et de **Deep Learning** afin de développer 
      un outil de **classification automatisée de cellules sanguines**, dont des cellules anormales, afin d'aider les praticiens lors de l'analyse des frottis sanguins de patients.
      """
    ) 


    st.markdown(
        """
        ## Quelques rappels d'hématologie
        """
      )
    cont_1 = st.container()
    with cont_1.expander('Les cellules du sang périphérique'):
      st.markdown(
      """
        Le sang périphérique humain contient de nombreux types cellulaires, principalement issus de la moelle osseuse et regroupés en trois catégories: 
        les érythrocytes, les leucocytes et les thrombocytes. 

        Les érythrocytes et les thrombocytes se distinguent morphologiquement des leucocytes par l'absence de noyau. 
        Les proportions moyennes des leucocytes chez les adultes sains sont de 60 à 70% de neutrophiles, 20 à 25% de lymphocytes, 3 à 8% de monocytes, 2 à 4% d'éosinophiles et 0,5 à 4% de basophiles, 
        mais ces pourcentages peuvent varier d'un individu à l'autre.
      """
      )
      left_co, cent_co,last_co = st.columns(3)
      with cent_co:
        st.image(
          Image.open("streamlit_app/assets/composants_sang_peripherique.jpg"), width=700, output_format='PNG',
          caption = "Figure 1 : Cellules unicellulaires présentes dans le sang " + 
            "périphérique"
        )
    
    cont_2 = st.container()
    with cont_2.expander('Leucémies Myéloïdes et Pro-myéloïdes Aiguës'):
      st.markdown(
        """

        La leucémie myéloïde aiguë (AML) est un type de cancer touchant les cellules précurseures de la lignée myéloïde, entravant leur différenciation 
        et causant leur sur-prolifération dans le sang périphérique. L'AML est une famille de cancers avec une importante diversité de mutations touchant 
        différents types de progéniteurs, tels que les myéloblastes, les myélocytes et les promyélocytes.
        """
      )
      left_co, cent_co,last_co = st.columns(3)
      with cent_co:
        st.image(
          Image.open("streamlit_app/assets/progeniteurs_hematopoietiques_aml_apl.jpg"), width=750, output_format='PNG',
          caption = "Figure 2 : Illustration simplifiée des progéniteurs " + 
            "hématopoïétiques touchés dans l'AML et l'APL"
          )
        

    st.markdown(
      """
      ## Les données
      Pour ce projet, nous avons collecté les images de frottis sanguins de 4 sources différentes :

      - Données Barcelona, patients sains (sources : [Acevedo et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S0169260719303578?via%3Dihub) ; [Boldú et al., 2021](https://www.sciencedirect.com/science/article/abs/pii/S0169260721000742?via%3Dihub))  
      - [Données Kaggle](https://www.kaggle.com/datasets/eugeneshenderov/acute-promyelocytic-leukemia-apl), patients malades AML et APL
      - [Données Munich](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958#61080958168cfd58d3c042fcbec1277728bd03e1) (source : [Matek, C. et al., 2019](https://www.nature.com/articles/s42256-019-0101-9))  
      - [Données Raabin](https://raabindata.com/free-data/#double-labeled-cropped-cells), patients sains uniquement (source : [Mousavi Kouzehkanan et al., 2021](https://www.biorxiv.org/content/10.1101/2021.05.02.442287v4))

      Les nomenclatures des types cellulaires étaient variées et différentes d'une source de données à une autre. 
      Il nous a fallu, dans un premier temps, trouvé une nomenclature commune. Ainsi, nous sommes passés de 48 noms de classes initiales à 15 seulement.
      """
    )
    
    df = pd.read_csv('data/BCC_dataset_final.csv', index_col = 0)
    df.head()
    cells_count = df[['source', 'blood_cell']].value_counts()
    cells_count = cells_count.rename_axis(['source', 'blood_cell']).reset_index(name='counts')
    
    fig_bar = px.bar(
        cells_count, 
        x = 'blood_cell',
        y = 'counts',
        color = 'source',
    )
    fig_bar.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
    
    fig_bar.update_traces(
        hovertemplate = '%{x}: %{y}'
    )

    fig_bar.update_layout(
        xaxis_title = 'Cellule sanguine',
        yaxis_title = 'Nombre d\'images',
    )

    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption('Figure 1 : Distribution des images par catégorie')

