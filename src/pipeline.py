#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
import pandas as pd
import sys
from pathlib import Path
ROOT = Path().resolve().parent
sys.path.insert(0, str(ROOT))
from src.filters import calculate_rca



def build_Mcp_matrix(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Input: 
    - Prende un dataframe con le seguenti colonne

    t: year
    i/j: exporter/importer
    k: product
    Mcp: valori binari di RCA
    

    Output:
    - Per ogni anno restituisce la matrice Mcp :

    RIGHE
    i/j: exporter / importer

    COLONNE
    k: product

    """

    df_year = df[df["t"] == year]

    matrix = df_year.pivot(index='j', columns='k', values='Mcp').fillna(0)

    return matrix



def Mcp_pipeline(
    df,
    country_col,
    product_col,
    value_col,
    year_col,
    output_dir
):

    """
    Input: 
    - Prende un file di dati BACI aggregati per import/expoty con le seguenti colonne:

    t: year
    i/j: exporter/importer
    k: product
    v: value

    

    Output:
    - Crea per ogni anno il file parquet con la matrice Mcp export/import
    
    """

    os.makedirs(output_dir, exist_ok=True)

    df = calculate_rca(df, country_col, product_col, value_col, year_col)

    matrices = {}

    for year in sorted(df[year_col].unique()):

        Mcp = build_Mcp_matrix(df, year)

        path = os.path.join(output_dir, f"Mcp_import_{year}_filtered.parquet")
        Mcp.to_parquet(path)

        matrices[year] = Mcp

    return matrices
    


    
    
   

