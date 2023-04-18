# Authors:
# Thomas Dumazert
# Hajer Souaifi-Amara
# Creation date: 23FEB2023 by Thomas
# Modification date: 21MAR2023 by Hajer
# Modification date: 23MAR2023 by Thomas
# Modification date: 29MAR2023 by Hajer

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Modélisation"
sidebar_name = "Modélisation"

def run():

    st.title(title)

    st.markdown("---")

    cont_0 = st.container()

    cont_0.markdown(
        """
        ## Criblage des couples modèle pré-entrainé / dataset

        Afin de cibler les pré-traitements à effectuer et les modèles à entraîner, nous avons voulu 
        répondre à trois questions :
        - Les CNN pré-entrainés sont-ils efficaces pour répondre à notre problématique ?
        - Quel est l’impact du rééquilibrage des classes sur les performances de classification ?
        - Le rognage préalable des images est-il bénéfique pour les performances des modèles ? 

        Pour cela, nous avons suivi un protocole de test systématique des couples modèle 
        pré-entraîné / dataset.

        """
    )

    with cont_0.expander('**Liste des modèles à tester**'):
        st.markdown(
            """
            Les modèles que nous avons testé sont :
            - Alexnet ;
            - DenseNet ;
            - Inception ;
            - ResNet ;
            - SqueezeNet ;
            - VGG.

            """
        )
    
    with cont_0.expander('**Liste des datasets à tester**'):
        st.markdown(
            """
            Les dataset que nous avons testé sont :
            - Le dataset original, nommé « original/unbalanced » ;
            - Un dataset équilibré sans préprocessing, nommé « original/balanced » ; 
            - Un dataset préprocessé mais non équilibré, nommé « preprocessed/unbalanced » ;
            - Un dataset préprocessé et équilibré, nommé « preprocessed/balanced ».

            L'équilibrage des datasets a été réalisé de façon à atteindre 3000 images par classe.
            Cela a été fait en augmentant le nombre d'images pour les classes sous-représentées et en 
            choisissant aléatoirement les images pour les classes sur-représentées.

            Le preprocessing a consisté en l'application des conclusions de la phase d'exploration 
            des données, à savoir :
            - rognage des images à une dimension de 360x360 quand cela est possible ;
            - compression des images sinon.
            """
        )
    
    with cont_0.expander('**Protocoles de test**'):
        st.markdown(
            """
            Chaque couple modèle pré-entraîné / dataset a été testé.

            Les dataset ont été séparé aléatoirement en trois ensembles :
            - Un ensemble d'entraînement comportant 70% des images pour chaque catégorie ;
            - Un ensemble de validation comportant 20% des images ;
            - un ensemble de test comportant 10% des images.

            Chaque modèle a été entrainé sur 30 epochs en utilisant les ensembles d'entraînement 
            et de validation.

            Aucun callback n'a été mis en place afin de s'assurer que tous les modèles 
            s'entrainent dans les mêmes conditions.

            """
        )

    with cont_0.expander('**Résultats**'):
        st.image(
            Image.open("data/models_results/pretrained_histories.png"),
            caption = "Figure 1 : Historiques d'entraînement"
        )

        st.markdown(
            """
            La Figure 1 permet de tirer trois conclusions :
            - l'utilisation d'un modèle pré-entraîné est pertinente pour notre problématique ;
            - l'adaptation préalable de la taille des images (notamment par rognage) ne semble pas 
            avoir d'impact significatif sur les performances du modèle.
            - l'équilibrage des classes a un effet sur l'accuracy du modèle. Ce qui était attendu 
            au regard de la répartition des images parmi les différentes classes.
            

            Toutefois, cette baisse de l'accuracy se fait-elle au bénéfice du score f1 ?.

            """
        )

        st.image(
            Image.open("data/models_results/pretrained_model_f1-score.png"),
            caption = "Figure 2 : F1-score pour les différents couples modèle/dataset"
        )

        st.markdown(
            """
            On remarque clairement sur la Figure 2 que le F1-score varie très peu selon le dataset. 
            
            Nous avons mis en relation ces variations avec les variations d'accuracy (cf. Figure 3).
            
            """
        )

        st.image(
            Image.open("data/models_results/pretrained_model_accuracy_vs_f1-score.png"),
            caption = "Figure 3 : Accuracy vs F1-score"
        )

        st.markdown(
            """
            Sur cette figure, nous avons l'accuracy en abscisse et le F1-score en ordonnée. On 
            constate que le F1-score varie bien moins que l'accuracy en fonction de l'équilibrage 
            ou non du dataset (même si ces variations sont minimes).

            On en conclue donc que l'équilibrage du dataset n'a pas d'impact bénéfique 
            suffisant sur les performances de classification des modèles pour être justifié.
            
            """
        )
    
    with cont_0.expander('**Conclusion**'):
        st.markdown(
            """
            Les modèles pré-entrainés sont-ils efficaces pour répondre à notre 
            problématique ? -> oui.  
            Un autre critère à prendre en compte est 
            la complexité du modèle ainsi que le temps d'entraînement nécessaire. En raison de 
            limitations techniques, nous avons été amenés à préférer un modèle simple et rapide 
            à entraîner mais tout de même efficace (VGG16) à un modèle plus complexe et plus efficace 
            mais plus long à entraîner (SqueezeNet).

            Quel est l'impact du rééquilibrage des classes sur les performances de 
            classification ? -> impact limité.  
            Même si cet impact varie en fonction du modèle choisi, un 
            équilibrage des classes de cellules par pondération sera préféré.

            Le rognage préalable des images est-il bénéfique pour les performances des 
            modèles ? -> non.  
            Le rognage préalable des images n'a qu'un effet très limité sur les performances des 
            modèles, quel que soit le modèle.
            
            """
        )


    cont_1 = st.container()
    cont_1.markdown(
        """
        ## Entrainement des modèles sélectionnés
        
        """
    )
   
    #Maintenant que nous avons établi quels modèles entraîner et avec quel dataset. Nous 
    #    nous passons à l'entraînement des modèles.

    with cont_1.expander('**Data preprocessing et data augmentation**'):
        st.markdown(
            """
            ###  Preprocessing

            Dans nos modèles, nous avons ajouté des **couches de prétraitement** :
            -	Resizing : pour redimensionner les images du lot au format 256x256 
            -	Centering and cropping : pour centrer et 'cropper' les images au format 180x180
            -	Rescaling : pour normaliser les pixels des images entre 0 et 1 

            
            Ainsi que des **couches d'augmentation** de données afin de se prémunir d'un éventuel sur-apprentissage lors de l'entraînement des modèles :
            -	RandomFlip retourne horizontalement et verticalement des images de manière aléatoire (certaines seront retourner d'autres non)
            -	RandomRotation pour faire pivoter une image de façon aléatoire sur une plage entre 0 et la valeur choisis (60 degrés)
            -	RandomWidth / RandomHeight sont des plages (en fraction de la largeur ou de la hauteur totale) à l'intérieur desquelles on peut redimensionner aléatoirement des images verticalement ou horizontalement.
            """
        )
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(
            Image.open("streamlit_app/data/models/Preprocessed_augmentated_images_example.png"),
            )

    with cont_1.expander('**Rééquilibrer les classes par la pondération**'):
        st.markdown(
            """
            ### Équilibrage par pondération

            Pour l'entraînement de nos modèles, nous avons utilisé la pondération des classes afin de corriger l'éventuel biais causé par le déséquilibre des catégories de cellules sanguines. 
            Nous avons utilisé une pondération qui attribue des poids inversement proportionels à la fréquence de la catégorie afin de favoriser les cellules en sous-effectif (dont les 'IG' font partie).
            """
        )
