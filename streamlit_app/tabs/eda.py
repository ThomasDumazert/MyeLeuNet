import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px

title = "Exploration des données"
sidebar_name = "Exploration des données"

def run():

    df = pd.read_csv('data/BCC_dataset_final.csv', index_col = 0)
    df.head()
    cell_cat_to_remove = ['UNS', 'UNP']
    labeled_df = df[~df['blood_cell'].isin(cell_cat_to_remove)]
    labeled_df['blood_cell'] = labeled_df['blood_cell'].replace(['IG', 'MMC', 'MYC', 'PYC'], 'IG')

    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        ### Distribution des images en fonction du type de cellules sanguines
        Nous avons commencé par étudier la distribution des images parmi les différentes classes.

        Nous avions à l'origine près de 40 classes différentes. Nous avons décidé de regrouper les types 
        cellulaires ayant une origine ou une fonction proche, afin de pouvoir travailler sur un 
        problème de classification plus simple et plus cohérent. 

        De plus, nous avons intégré à la catégorie 'IG' (immature granulocyte) les catégories: IG, MMC 
        (métamyélocyte), MYC (myélocyte), PMY (promyélocyte). Ce sont les types cellulaires en cause 
        dans la leucémie.

        Malgré ces regroupements, la distribution des catégories reste fortement déséquilibrée.
        """
    )
    col1, col2=st.columns(2)
    with col1:
        
        cells_count = labeled_df[['source', 'blood_cell']].value_counts()
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
        st.caption('Figure 1 : Distribution des cellules sanguines (10 catégories)')

    with col2:
        # Create a pivoted DataFrame with a total row and a total column
        pivot_cells_count = cells_count.pivot(index='blood_cell', columns='source', values='counts')
        pivot_cells_count['Total'] = pivot_cells_count.apply(lambda r: r.sum(), axis=1)
        total_row = pd.DataFrame({col: pivot_cells_count[col].sum() for col in pivot_cells_count.columns}, index=['Total'])
        pivot_cells_count = pd.concat([pivot_cells_count, total_row], axis=0)

        st.dataframe(pivot_cells_count, use_container_width=True, height=493)
        st.caption('Tableau 1 : Répartition des images par cellule sanguine et par source')

    st.markdown(
        """
        ### Analyse des images

        """
         )
 

    cont_1 = st.container()
    with cont_1.expander('Taille des images'):
        left_co, cent_co,last_co = st.columns(3)
        with left_co:
            sizes = labeled_df.groupby(['height', 'width', 'source']).agg(
            count=('image_path', 'count'),
            log_count=('image_path', lambda x: np.log2(x.count()))
            ).reset_index()

            fig_sizes = px.scatter(
            sizes, x='height', y='width',size='log_count',
            opacity=0.9, color='source', hover_name='source',
            hover_data={'source': False,
                        'height': True,
                        'width': True,
                        'log_count': False,
                        'count': True
                        },
            template='seaborn')
            st.plotly_chart(fig_sizes)
            st.caption('Figure 2 : Dimensions des images suivant la source de données')
        with last_co:
            st.markdown(
                """
                Les dimensions des images varient entre 360x359 à 575x575. 

                Une étape de "standardisation" des images sera nécessaire en pré-traitement.

                """
            ) 

    st.markdown(
        """
        #### Recherche d'outliers

        """
    ) 

    luminance, brightness, black_pixels = st.tabs(['Selon la luminance des images', 'Selon la luminosité des images', 'Selon la proportion de pixels noirs'])
    
    with luminance: 
        col1, col2=st.columns(2)
        with col1:
            st.image(
                Image.open("data/eda/Luminance_boxplots.png"),
                caption = "Figure 3 : Box-plots des luminances"
            )
        with col2: 
            st.image(
                Image.open("data/eda/Outliers_Luminance.png"),  width=500, output_format='PNG',
                caption = "Figure 4 : Exemples d'ouliers selon la luminance"
            )
    with brightness: 
        col1, col2=st.columns(2)
        with col1: 
            st.image(
                Image.open("data/eda/Brightness_boxplots.png"),
                caption = "Figure 5 : Box-plots des luminosités"
            )
        with col2: 
            st.image(
                Image.open("data/eda/Outliers_Brightness.png"), width=500, output_format='PNG',
                caption = "Figure 6 : Exemples d'ouliers selon la luminosité"
            )

    with black_pixels:
        col1, col2=st.columns(2)
        with col1:
            st.image(
                Image.open("data/eda_results/gray_pixels_boxplot_by_pixels_range.png"),
                caption = "Figure 7 : Proportions de pixels par tranche de valeus de gris"
            )
            st.markdown(
                """
                La Figure 7 montre que certaines images présentes des proportions anormalement élevées
                de pixels très sombres (en-dessous de 30) .
                """
            )
        with col2:
            st.image(
                Image.open("data/eda_results/number_of_images_vs_proportion_of_black_pixels.png"),
                caption = "Figure 8 : Nombre d'images par rapport à la proportion de pixels noirs"
            )
            st.markdown(
                """
                La Figure 8 montre qu'au-delà de 5% de pixels noirs (niveau de 
                gris en-dessous de 30), on ne trouve presque plus aucune image.
                """
            )

    st.markdown(
        """
        Nous nous sommes finalement basés uniquement sur la luminance et la luminosité des images 
        pour écarter les outliers, ce qui nous a permis d'éliminer 178 images atypiques.
        
        """
    )    
        
    st.markdown(
        """
        ### Qualité et colorimétrie moyenne des images
        """
        )
    
    cont_1 = st.container()
    with cont_1.expander('Qualité des images par source et par type cellulaire'):
        st.image(
            Image.open("data/eda_results/gray_pixels_distribution_by_source-blood_cell.png"),
            caption = "Figure 9 : Distribution des valeurs des pixels (nuances de gris) en fonction de la cellule sanguine représentée"
        )

        st.markdown(
            """
            La Figure 9 montre qu'au sein d'un même dataset, toutes les images semblent avoir 
            une répartition des valeurs des pixels similaires, malgré quelques anomalies.

            """
        )

    cont_2 = st.container()
    with cont_2.expander('Représentation de cellules "moyennes" et histogrammes des canaux RGB'):
        NEU, LYM, MYB, MON, EOS, IG, MYC, MMC, PYC, ART, PLT, ERY, BAS = st.tabs(['NEU', 'LYM', 'MYB', 'MON', 'EOS', 'IG', 'MYC', 'MMC', 'PYC', 'ART', 'PLT', 'ERY', 'BAS'])
        with NEU:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_NEU.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_NEU.png"), width=500
                )       
        with LYM:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_LYM.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_LYM.png"), width=500
                )       
        with MYB:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_MYB.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_MYB.png"), width=500
                )     
        with MON:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_MON.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_MON.png"), width=500
                )       
        with EOS:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_EOS.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_EOS.png"), width=500
                )    
        with IG:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_IG.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_IG.png"), width=500
                ) 
        with MYC:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_MYC.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_MYC.png"), width=500
                ) 
        with MMC:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_MMC.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_MMC.png"), width=500
                )
        with PYC:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_PYC.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_PYC.png"), width=500
                ) 
        with ART:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_ART.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_ART.png"), width=500
                ) 
        with PLT:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_PLT.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_PLT.png"), width=500
                ) 
        with ERY:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_ERY.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_ERY.png"), width=500
                ) 
        with BAS:
            col1, col2=st.columns(2)
            with col1: 
                st.image(
                    Image.open("data/eda/avg_img_BAS.png"), width=350
                )
            with col2: 
                st.image(
                    Image.open("data/eda/avg_hist_BAS.png"), width=500
                ) 

    st.markdown(
        """
        ### Analyse de la position des cellules dans les images
        
        Comme la dimension des images présente une grande disparité, il sera indispensable de les 
        redimensionner. Toutefois, nous souhaitions procéder à ce redimensionnement avec le 
        moins de perte de données possible, idéalement en rognant une partie du fond des images.

        Nous avons donc vérifié si cela était faisable pour toutes les images en identifiant
        la position moyenne des cellules sur les images ainsi que leur 
        taille.
        """
    )

    cont = st.container()
    with cont.expander('Analyse de la position moyenne des cellules'):

        st.image(
            Image.open("data/eda_results/mean_blood_cell_position_by_source-blood_cell.png"),
            caption = "Figure 10 : Analyse de la position moyenne des cellules"
        )

        st.markdown(
            """
            On retrouve sur la Figure 10, surimposés à l'image moyenne de la cellule selon la 
            provenance des images, le rectangle délimitant la cellule en **:red[rouge]** ; le rectangle de 
            rognage, en **:green[vert]**, calculé à partir du rectangle rouge en utilisant un facteur 2 afin de 
            nous assurer de ne pas tronquer la cellule et de conserver suffisamment de contexte ; et 
            enfin le rectangle **:blue[bleu]** de 360x360 centré sur le centre du rectangle rouge.

            On peut en tirer plusieurs observations :
            - comme attendu, il y a d'importantes variations de taille des cellules sanguines selon 
            leur type mais aussi la provenance des images ;
            - la taille théorique des images rognées serait de 572x582, ce qui correspond à la 
            plus grande taille de rognage calculée (basophiles « bas » de Raabin) ;
            - la méthode choisie pour détecter la cellule ne donne pas systématiquement des résultats 
            satisfaisants ;
            - toutefois, en fixant la taille des images à 360x360, cela permettra, dans la majorité 
            des cas, de rogner les images sans perte d'information ;
            - en revanche, les basophiles, les éosinophiles et les monocytes de Raabin, ainsi que 
            pour les neutrophiles de Munich, ne pourront pas être rognés.

            """
        )