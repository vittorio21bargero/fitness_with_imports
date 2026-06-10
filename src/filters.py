#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np



def calculate_rca(
    df: pd.DataFrame,
    country_col: str,
    product_col: str,
    value_col: str,
    year_col: str
) -> pd.DataFrame:
    
    """
    
    Input: 
    - Prende un dataframe con i dati aggregati annuali di export/import per prodotto per ogni nazione

    t: year
    i/j: exporter / importer
    k: product
    v: value
    
    Output:
    - Calcola i valori di Mcp  


    """

    df = df.copy()

    # per ogni anno ( è sempre presente groupby[year] ) calcolo:

    # export/import totale per ogni nazione
    country_tot = df.groupby([year_col, country_col])[value_col].transform('sum')

    # export/import totale globale per il prodotto
    product_tot = df.groupby([year_col, product_col])[value_col].transform('sum')

    # export/import totale globale
    total = df.groupby(year_col)[value_col].transform('sum')

    # valore CONTINUO di RCA
    df['RCA'] = (df[value_col] / country_tot) / (product_tot / total)

    # valore BINARIO di RCA (Mcp)
    df['Mcp'] = (df['RCA'] >= 1).astype(int)

    return df

def compute_X(C, M_export):
    """
    X_cp = (C @ M_export)_cp
    Questa matrice mi dice a quanti output p' esportati dal paese c serve ogni prodotto p
    
    """
    return C @ M_export


def compute_N(X, M_import):

    """
    Questa matrice con element-wise product mi permette di selezionare solo i prodotti importati 
    
    """
   
    return X * M_import



