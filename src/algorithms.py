#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
import re  # per estrarre l'anno dal nome file
from scipy.sparse.linalg import eigs


# In[4]:

# calcolo  ECI 

def compute_eci(Mcp: pd.DataFrame):
    """
    Calcola ECI e PCI usando il metodo dell'autovettore.

    Parametri:
    ----------
    Mcp : pd.DataFrame
        Matrice binaria paese-prodotto 

    Returns
    -------
    eci : pd.Series
        Economic Complexity Index dei paesi
    pci : pd.Series
        Product Complexity Index dei prodotti
    """

    M = Mcp.values.astype(float)

    # diversity and ubiquity
    kc = M.sum(axis=1)  # diversity countries
    kp = M.sum(axis=0)  # ubiquity products

    # Evita divisioni per zero
    kc[kc == 0] = 1
    kp[kp == 0] = 1

    # Matrice normalizzata paese-paese
    M_tilde = (M / kc[:, None]) @ (M.T / kp[:,None ])
    #[   ] il None all' interno delle parentesi mi permette di rendere, quello che prima era un vettore riga/colonna,
    #un vettore colonna/riga in modo che le operazioni con le matrici vengano eseguite correttamente 

    # Autovalori e autovettori (tramite la funzione eig)
    eigvals, eigvecs = np.linalg.eig(M_tilde)

    # Ordina per autovalore decrescente
    idx = np.argsort(eigvals)[::-1]
    # idx assegna a ciascun autovalore il suo posizionamento del ranking in ordine crescente (valore 0 al più piccolo e così via)
    eigvals = eigvals[idx] # riordino gli autovalori (sempre ordine crescente)
    eigvecs = eigvecs[:, idx] #riordino allo stesso modo gli autovettori

    # Prendo il secondo autovettore 
    eci_raw = np.real(eigvecs[:, 1])

    # Standardizzazione (media 0, varianza 1)
    eci = (eci_raw - np.mean(eci_raw)) / np.std(eci_raw)

    # --- PCI ---
    # Proiezione sui prodotti
    pci_raw = (M.T / kc[None, :]) @ eci

    pci = (pci_raw - np.mean(pci_raw)) / np.std(pci_raw)

    # Converte in pandas Series
    eci = pd.Series(eci, index=Mcp.index, name="ECI")
    pci = pd.Series(pci, index=Mcp.columns, name="PCI")

    return eci, pci


    #pd.Series mi permette di assgnare a ciascuno dei valori di ECI il paese a cui corrisponde
    #altrimenti avrei un array di cui non capisco a chi appartiene uno specifico valore



# calcolo EFC 


def compute_fitness(Mcp: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-6):

    """
    Calcola Fitness e Complessità dei prodotti (Tacchella).

    Parametri:
    ----------
    Mcp : pd.DataFrame
        Matrice binaria paese-prodotto (righe = paesi, colonne = prodotti)
    max_iter : int
        Numero massimo di iterazioni
    tol : float
        Tolleranza per convergenza

    Returns
    -------
    fitness : pd.Series
        Fitness dei paesi
    complexity : pd.Series
        Complessità dei prodotti
    """

    M = Mcp.values.astype(float)
    n_c, n_p = M.shape
    # numero di righe (countries) e colonne (products)

    # inizializzazione vettori: assegno complessità 1 a tutti i countries e products
    F = np.ones(n_c)
    Q = np.ones(n_p)

    for _ in range(max_iter):
        # salvo i valori dello step precedente: per poter poi verificare la convergenza 
        F_old = F.copy()
        Q_old = Q.copy()

        # aggiornamento Fitness
        F = M @ Q

        # aggiornamento Complessità (non lineare!)
        Q = 1 / (M.T @ (1 / F))

        # normalizzazione (evita divergenze numeriche)
        F = F / np.mean(F)
        Q = Q / np.mean(Q)

        # controllo convergenza
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break

    fitness = pd.Series(F, index=Mcp.index, name="Fitness")
    complexity = pd.Series(Q, index=Mcp.columns, name="Complexity")
    
    

    return fitness, complexity, convergence_steps


# In[5]:


def compute_modified_fitness_1(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    phi,
    theta,
    max_iter: int = 1000,
    tol: float = 1e-6
):
    """
    Calcola Fitness modificata con valore aggiunto e matrice di composizione.

    Parametri:
    ----------
    Mcp : pd.DataFrame
        Matrice paese-prodotto (binaria o pesata)
    C : pd.Dataframe
        Matrice di composizione prodotti-prodotti (n_p x n_p), nilpotente
    phi:
    theta:

    max_iter : int
        Numero massimo iterazioni
    tol : float
        Tolleranza per convergenza

    Returns
    -------
    fitness : pd.Series
    complexity : pd.Series

    """

    M = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape
    
    

    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)
    
    # 2) SISTEMA DINAMICO
    for _ in range(max_iter):

        # salvataggio valori del passo precedente
        F_old = F.copy()
        Q_old = Q.copy()

        # equazione 1
        F = phi(M , Q )

        # equazione 2
        Delta = theta(F, M )
        CQ = C @ Q          # contributo input
        Q = CQ + Delta      # update totale
        
        
        # 3) NORMALIZZAZIONE
        F = F / np.mean(F)
        CQ = CQ / np.mean(Q)
        Delta = Delta / np.mean(Q)
        Q = Q / np.mean(Q)
        
    

        # convergenza
        if (
            np.all(np.abs(F - F_old) < tol) and
            np.all(np.abs(Q - Q_old) < tol) 
            
        ):
            break

    fitness = pd.Series(F, index=Mcp.index, name="Fitness")
    CQ = pd.Series(CQ, index=Mcp.columns, name="input_complexity")
    Delta = pd.Series(Delta, index=Mcp.columns, name="intrinsic_complexity")
    complexity = pd.Series(Q, index=Mcp.columns, name="total_complexity")
    
    
    return  complexity, fitness, CQ, Delta


def compute_modified_fitness_2(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    phi,
    theta,
    max_iter: int = 1000,
    tol: float = 1e-6
):
    """
    Calcola Fitness modificata con valore aggiunto e matrice di composizione.

    Parametri:
    ----------
    Mcp : pd.DataFrame
        Matrice paese-prodotto (binaria o pesata)
    S : pd.Dataframe
        S = I + C + C^2 + ... + C^k
    phi:
    theta:

    max_iter : int
        Numero massimo iterazioni
    tol : float
        Tolleranza per convergenza

    Returns
    -------
    fitness : pd.Series
    complexity : pd.Series

    """

    M = Mcp.values.astype(float)
    S = S.values.astype(float)
    n_c, n_p = M.shape
    
    

    # 1) INIZIALIZZAZIONE
    Q = np.ones(n_p)
    F = np.ones(n_c)
    
    # 2) SISTEMA DINAMICO
    for _ in range(max_iter):

        # salvataggio valori al passo precedente
        F_old = F.copy()
        Q_old = Q.copy()

        # equazione 1
        F = phi(M , Q )

        # equazione 2
        Delta = theta(F, M )
        Q = S @ Delta
       
        
        # 3) NORMALIZZAZIONE
        F = F / np.mean(F)
        Q = Q / np.mean(Q)
        
    

        # convergenza
        if (
            np.all(np.abs(F - F_old) < tol) and
            np.all(np.abs(Q - Q_old) < tol) 
            
        ):
            break

    #fitness = pd.Series(F, index=Mcp.index, name="Fitness")
    #CQ = pd.Series(CQ, index=Mcp.columns, name="input_complexity")
    #Delta = pd.Series(Delta, index=Mcp.columns, name="intrinsic_complexity")
    complexity = pd.Series(Q, index=Mcp.columns, name="total_complexity")
    
    return  complexity     #, CQ, Delta fitness,

def compute_modified_fitness_imports(
    M_E: pd.DataFrame,
    M_I: pd.DataFrame,
    C: pd.DataFrame,
    phi,
    theta,
    M: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-6,
):
    M_E_arr = M_E.values.astype(float)
    M_I_arr = M_I.values.astype(float)
    C_arr   = C.values.astype(float)

    n_c, n_p = M_E_arr.shape

    # Tensori
    C_tensor            = build_C_tensor(C_arr, M)
    M_I_tensor, M_E_tensor = build_filter_tensors(M_I_arr, M_E_arr, M)

    # Inizializzazione
    F     = np.ones(n_c)
    Q     = np.ones(n_p)
    Delta = np.ones((n_c, n_p, M))

    for _ in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()

        # 2.1 Country fitness
        penalty = compute_penalty(Delta, M_I_tensor, M_E_tensor, C_tensor)
        F = phi(M_E=M_E_arr, Q=Q_old, penalty=penalty)

        # 2.2 Complessità intrinseca
        Delta_new = theta(F=F, M=M_E_arr)

        # 2.3 Complessità totale
        CQ = C_arr @ Q
        Q  = CQ + Delta_new

        # Normalizzazione
        mean_Q    = np.mean(Q)
        F         = F / mean_Q  # o normalize(F) se phi non normalizza
        CQ        = CQ / mean_Q
        Delta_new = Delta_new / mean_Q   # ← normalizza qui, una volta sola
        Q         = Q / mean_Q

        # Update Delta (ora Delta_new è già normalizzato, shift_and_scale NON deve ridivdere)
        Delta = shift_and_scale_delta(Delta, Delta_new, mean_Q=1.0)

        # Convergenza
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break

    fitness           = pd.Series(F,         index=M_E.index,   name="Fitness")
    complexity        = pd.Series(Q,         index=M_E.columns, name="Complexity")
    input_complexity  = pd.Series(CQ,        index=M_E.columns, name="Input Complexity")
    intrinsic_series  = pd.Series(Delta_new, index=M_E.columns, name="Intrinsic Complexity")
   
    return complexity, fitness, input_complexity, intrinsic_series

def run_fitness_tracked(Mcp, max_iter=1000, tol=1e-6):
    M = Mcp.values.astype(float)
    n_c, n_p = M.shape
    F, Q = np.ones(n_c), np.ones(n_p)
    history = []
    for i in range(max_iter):
        F_old, Q_old = F.copy(), Q.copy()
        F = M @ Q
        Q = 1 / (M.T @ (1 / F))
        F = F / np.mean(F)
        Q = Q / np.mean(Q)
        history.append((np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old))))
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break
    return np.array(history), i

def run_modified_fitness_tracked(Mcp, C, phi, theta, max_iter=1000, tol=1e-6):
    M = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape
    F, Q = np.ones(n_c), np.ones(n_p)
    history = []
    for i in range(max_iter):
        F_old, Q_old = F.copy(), Q.copy()
        F     = phi(M, Q)
        Delta = theta(F, M)
        CQ    = C @ Q
        Q     = CQ + Delta
        F     = F / np.mean(F)
        CQ    = CQ / np.mean(Q)
        Delta = Delta / np.mean(Q)
        Q     = Q / np.mean(Q)
        history.append((np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old))))
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break
    return np.array(history), i


# In[ ]:




