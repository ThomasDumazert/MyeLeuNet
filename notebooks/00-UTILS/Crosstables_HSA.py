###################################################################
#Author : **Hajer SOUAIFI-AMARA**

#Creation date : 04-MAR-2023

#Final date : 17-MAR-2023

#Modification date : 17-MAR-2023 - transformation en .py

##################################################################


#################################################################
#                    Crosstables                                #
#################################################################


import pandas as pd
import numpy as np
import sys, os
from pathlib import Path

from sklearn.model_selection import train_test_split


#####################################################################
### Fonction print_crosstables: qui affiche les tableaux croisés

def print_crosstables(df):
    
    '''
    Args:
    - df: dataframe with a column named 'blood_cell' and a column 'source'
    '''
    
    crosstabl = pd.crosstab(index=df['blood_cell'], columns=df['source'])
    pourcentages = crosstabl.apply(lambda x: 100 * x / float(x.sum()), axis=1)
    crosstabl['Total'] = crosstabl.sum(axis=1)
    crosstabl['% Total'] = 100 * crosstabl['Total'] / crosstabl['Total'].sum()
    for col in crosstabl.columns[:-2]:
        crosstabl[col + ' %'] = 100 * crosstabl[col] / crosstabl['Total']
    
    return display(crosstabl)