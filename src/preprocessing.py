#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd


def process_BACI_raw_folder(input_folder: str, output_path: str):
    """
    Input: 
    - Prende una cartella (input_folder)
    - Va a vedere i file BACI csv all' interno della cartella che hanno le seguenti colonne:

    t: year
    i: exporter
    j: importer
    k: product
    v: value

    

    Output:
    - Crea un UNICO file parquet con i dati aggregati annuali di export/import per prodotto per ogni nazione
    - Il file viene salvato nel percorso output_path

    t: year
    i/j: exporter / importer
    k: product
    v: value
    
    """
    
    all_dfs = []  # lista (vettore) con all' interno i vari dataframes che sono presenti nella cartella
    
    for filename in os.listdir(input_folder):
        # restituisce tutti i file csv della cartella input_folder
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)
            # ricostruisce il percorso completo del file: mi serve perchè devo ricostruirmi ancora il percorso completo
            
            # Legge i file selezionati sopra
            df = pd.read_csv(filepath)
            
            # Estrai anno dal nome file, es: "BACI_HS92_Y2024_V202601.csv"
            match = re.search(r'Y(\d{4})', filename)
            # va a controllare su tutti i file se è presente una Y seguita da 4 cifre
            # non è booleana ma si comporta in modo simile: 
            # se trova un oggetto nell'if corrisponde a un true al contrario a un false
            if match:
                year = int(match.group(1))
                # restituisce la parte dentro la parentesi all'interno di re.search
            else:
                print(f"Anno non trovato in {filename}, salto file")
                continue
            
            # Aggiungi colonna anno
            df['t'] = year
            
            
            # Aggrega (i/j, k, t) e somma v
            df_agg = (
                df_clean
                .groupby(['j', 'k', 't'], as_index=False)['v']
                .sum()
            )
            
            all_dfs.append(df_agg)
    
    # Unisci tutti i DataFrame
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Salva in parquet
    final_df.to_parquet(output_path, index=False)
    

    return final_df



def AIPNET_to_C(df, upstream_product_column_name, downstream_product_column_name, N):

    """
    - Questa funzione mi permette di convertire i dati AIPNET in una matrice di composizione binaria C_pp' (p è input di p')
    - Da applicare a ciascuno dei 10 database AIPNET (4/6-digits x HS02/07/12/17/22)
     
    Input:
    - M: pd.dataframe 
      Dataframe con i dati AIPNET 
    - upstream_product_column_name : nome della colonna coi prodotti upstream
    - downstream_product_column_name: nome della colonna coi prodotti downstream
    - N: numero di cifre della classificazione HS che voglio

    Output:
    - C: matrice di composizione
    
    """
    
    # creo una copia  
    C = df.copy()
    
    # converto gli indici delle due colonne in stringhe a n cifre (nel caso fillo con lo 0 davanti)
    C[upstream_product_column_name] = C[upstream_product_column_name].astype(str).str.zfill(N)  
    C[downstream_product_column_name] = C[downstream_product_column_name].astype(str).str.zfill(N)
    
    # creo la matrice di adiacenza (il numero di righe e colonne sarà diverso in quanto non vengono considerati prodotti puramente upstream/downstream)
    C = pd.crosstab(C[upstream_product_column_name], C[downstream_product_column_name])
    
    # aggiungo anche prodotti puramente upstream/downstream
    all_codes = sorted(set(C.index) | set(C.columns))
    C = C.reindex(index=all_codes, columns=all_codes, fill_value=0)

    return C


def BACI_HS6_to_HS4(df):

    """
    Input: 
    - File BACI HS6 (formato parquet) aggregato avente colonne:
    t: year
    i/j: exporter/importer
    k: product
    v: value

    Output: 
    - File BACI HS4 (formato parquet) aggregato avente colonne:
    t: year
    i/j: exporter/importer
    k: product
    v: value
    
    """

    # prendo le prime 4 cifre dei codici nella colonna "k"
    df['k'] = df['k'].str[:4]
    # aggrego anche i dati di export/import
    df = df.groupby(['j','t', 'k'])['v'].sum().reset_index()

    return df


def C_test(df):
    
    """
    Verifica che la matrice C abbia correttamente gli stessi codici su righe e colonne
    Se tutto è corretto dovrebbe dare 0

    Input:
    - Matrice C

    Output:
    - Numero di codici diversi tra righe e colonne
    
    """

    # Codici diversi riga e colonna
    C_codes_row = set(pd.unique(C.index.astype(str)))
    C_codes_columns = set(pd.unique(C.columns.astype(str)))
    missing_total =  C_codes_row ^ C_codes_columns

    return missing_total

    
    
def M_and_C_common_products(M, C):
    """
    Prima di runnare questa funzione è molto utile usare C_test per una prima verifica
    
    Input:
    - matrice Cpp'
    - matrice Mcp

    Output:
    - elementi diversi tra le due matrici
    
    """

    # estrazione codici prodotto HS + visualizzazione numero di codici 
    M_codes=set(pd.unique(M.columns.astype(str)))
    print("Numero di colonne Mcp: ",len(M_codes))
    C_codes = set(pd.unique(C.columns.astype(str)) )
    print("Numero di righe Cpp': ",len( C_codes))

    
    common_products = C_codes & M_codes
    print("prodotti in comune tra righe di Cpp e Mcp:",len(common_products))
    
    C_missing =  M_codes - C_codes
    print ("Prodotti presenti in M ma non in C:",C_missing)
    print ("Numero di Prodotti presenti in M ma non in C:",len(C_missing))
    M_missing = C_codes - M_codes
    print ("Numero di Prodotti presenti in M ma non in C:",M_missing)
    print ("Prodotti presenti in C ma non in M:",len(M_missing))
    

    return C_missing, M_missing

def count_number_of_0s_and_1s(C):
    
    C = np.array(C)

    count_0s = np.sum(C == 0)
    count_1s = np.sum(C == 1)

    return count_0s, count_1s
    

def missing_codes_economic_impact(missing, BACI_df ):
    """
    Input:
    - missing: lista di codici presenti in BACI ma non in AIPNET
    - BACI_df: dataframe con i dati di export aggregati

    Output:
    - Impatto economico dei codici non presenti in AIPNET 

    """
    
    # filtro i prodotti di interesse
    
    BACI_df_diff = BACI_df[BACI_df["k"].isin(missing)]  # isin() --- true or false   controlla quali righe hanno quei codici prodotto

    # export totale dei codici selezionati
    export_diff = BACI_df_diff["v"].sum()

    # export totale complessivo
    export_totale = BACI_df["v"].sum()
 
    # quota percentuale
    quota = export_diff / export_totale

    print("Export codici selezionati:", export_diff)
    print("Export totale:", export_totale)
    print("Quota:", quota)

    return export_diff, export_totale, quota



def autovettore_principale(df):
    """
    Calcola l'autovettore principale (associato al       massimo autovalore in modulo)

    Input:
    - df: pd.DataFrame (matrice quadrata)

    Output:
    - pd.Series (autovettore principale normalizzato)
    """
    
    A = df.values  # converto in numpy
    
    # autovalori e autovettori
    eigvals, eigvecs = np.linalg.eig(A)
    
    # indice autovalore dominante (modulo massimo)
    idx = np.argmax(np.abs(eigvals))
    
    # autovettore corrispondente
    v = eigvecs[:, idx]
    
    # normalizzazione (opzionale ma utile)
    v = np.real(v)  # evita complessi numerici
    v = v / np.linalg.norm(v)
    
    # ritorno come Series con stessi indici
    return pd.Series(v, index=df.index)