#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# 0) VECCHIO ALGORITMO (2022_HS4_analysis)

def compute_extended_fitness_April(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    max_iter: int = 100000,
    tol: float = 1e-15,
    track: bool = True,
    save_delta: bool = False,
    delta_window: int | None = None,
):
   

    # Gestione INPUT
    M  = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape
    
    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)
    
    # Salvataggio dei Delta ai passi precedenti
    history    = [] if track else None
    delta_hist = deque(maxlen=delta_window) if save_delta else None
    converged  = False
    
    
    
    i = 0
    for i in range(max_iter):
       
        F_old = F.copy()
        Q_old = Q.copy()
        
        
        #  2.1) EQUAZIONI SISTEMA DINAMICO
        F     = M @ Q_old                       
        Delta = 1.0 / (M.T @ (1.0 / F_old))
        mean_Delta = np.mean(Delta)
        Delta = Delta / mean_Delta
        CQ    = alpha * C @ Q_old                      
        Q     = CQ + Delta                   
       
        #  2.2) NORMALIZZAZIONE
        mean_Q = np.mean(Q)
        
        F     = F / np.mean(F)
        CQ    = CQ / mean_Q
        Delta = Delta /  mean_Q
        Q     = Q / mean_Q
        
        
        
        if track:
            history.append((np.mean(np.abs(F - F_old)), np.mean(np.abs(Q - Q_old))))
        if save_delta:
            for d in delta_hist:
                d /= meanQ
            delta_hist.append(Delta.copy())
        
        
        
        # 3) CONVERGENZA
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            converged = True
            
            
            break
    fitness    = pd.Series(F,     index=Mcp.index,   name="Fitness")
    CQ         = pd.Series(CQ,    index=Mcp.columns, name="input_complexity")
    Delta      = pd.Series(Delta, index=Mcp.columns, name="intrinsic_complexity")
    complexity = pd.Series(Q,     index=Mcp.columns, name="total_complexity")
    
    
    if track or save_delta:
        info = {"n_iter": i, "converged": converged}
        if track:
            info["history"] = np.array(history)
        if save_delta:
            # shape (n_salvate, n_p): righe = iterazioni, dalla più vecchia tenuta alla più recente
            info["delta_history"] = np.array(delta_hist)
        return complexity, fitness, CQ, Delta, info
    
    
    return complexity, fitness, CQ, Delta


# 2.1.1.0) FITNESS con C (à la Vito con i delta)

# matrice C: SI
# delta: SI
# import: NO

def compute_extended_fitness_with_delta(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    phi,
    theta,
    delta: float = 1e-2,      # <-- ecco il parametro, molto minore di 1
    max_iter: int = 1000,
    tol: float = 1e-6
):
    M = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape

    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)

    # storia della convergenza step-by-step
    # storia della convergenza step-by-step
    history = {"F_diff": [], "Q_diff": [], "F_mean_diff": [], "Q_mean_diff": []}

    # 2) SISTEMA DINAMICO
    for it in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()

        # equazione 1 + regolarizzazione delta
        F = phi(M, Q) + delta

        # equazione 2 + regolarizzazione delta
        Delta = theta(F, M)
        CQ = C @ Q                  # contributo input
        Q = CQ + Delta + delta      # update totale + delta


        # registro la variazione massima a questo step
        f_diff = np.max(np.abs(F - F_old))
        q_diff = np.max(np.abs(Q - Q_old))
        history["F_diff"].append(f_diff)
        history["Q_diff"].append(q_diff)
        history["F_mean_diff"].append(np.mean(np.abs(F - F_old)))
        history["Q_mean_diff"].append(np.mean(np.abs(Q - Q_old)))

        # convergenza
        if f_diff < tol and q_diff < tol:
            break

    fitness    = pd.Series(F,     index=Mcp.index,   name="Fitness")
    CQ         = pd.Series(CQ,    index=Mcp.columns, name="input_complexity")
    Delta      = pd.Series(Delta, index=Mcp.columns, name="intrinsic_complexity")
    complexity = pd.Series(Q,     index=Mcp.columns, name="total_complexity")

    return complexity, fitness, CQ, Delta, history

 

def peso_k_ordini(
    complexity,
    delta_history,
    C: pd.DataFrame,
    alpha: float = 0.5,
    delta_window: int = 5,
):
    """
    Calcola quanto incide in media ogni ordine k sulla complessità totale dei prodotti.

    Input:
    - complexity: Series/array con la total_complexity calcolata dall'algoritmo
    - delta_history: matrice (n_salvate, n_p) con i Delta alle iterazioni precedenti
    
    - C: matrice di composizione (convenzione Cp'p, p' input di p)
    - alpha: parametro di sconto
    - delta_window: numero di ordini da calcolare (deve essere <= n_salvate in delta_history)

    Output:
    - tabella_pesi: DataFrame con colonne "ordine" e "peso_medio"
    
    """
    Delta_matrix = np.zeros((len(C.index), delta_window))
    C = C.values.astype(float) 
    C_T = C.T  # trasposta, coerente con la direzione input->output

    
    for k in range(delta_window):  # k = 0,1,...,delta_window-1
        Delta_k_passi_fa = delta_history[-(k + 1)]  # -(k+1):devo partire dall'ultima riga dove ho il valore piu recente
        Delta_matrix[:, k] = (alpha ** k) * np.linalg.matrix_power(C_T, k) @ Delta_k_passi_fa

    peso_medio_ordine = Delta_matrix.mean(axis=0) 

    tabella_pesi = pd.DataFrame({
        "ordine": range(delta_window),
        "peso_medio": peso_medio_ordine
    })
    # aggiungo anche il peso cumulativo alla tabella
    tabella_pesi["peso_cumulativo"] = tabella_pesi["peso_medio"].cumsum()

    return tabella_pesi

def autovettore_principale(C_df):
    """
    Calcola l'autovettore principale (corrispondente all'autovalore
    di modulo massimo) di una matrice quadrata passata come DataFrame.

    Input:
    - df: matrice quadrata ( C.T nel nostro caso )

    Output:
    - autovettore_principale
    - rho = è il modulo dell'autovalore principale, ovvero il raggio spettrale della matrice C

    NOTA BENE:
    La matrice di cui devo calcolare è la TRASPOSTA di C
    
    """
    matrice = C_df.values  # converto il DataFrame in array numpy

    autovalori, autovettori = np.linalg.eig(matrice)

    # trovo l'indice dell'autovalore con modulo massimo
    idx = np.argmax(np.abs(autovalori))

    rho = np.abs(autovalori[idx])   # mi interessa il modulo dell'autovettore per rho
    autovettore_principale = np.real(autovettori[:, idx])

    # Fix del segno: forzo che la somma sia positiva
    if np.sum(autovettore_principale) < 0:
        autovettore_principale = -autovettore_principale

    return rho, autovettore_principale



def link_rewiring(C: pd.DataFrame, eta: float, seed: int | None = None):
    """
    Prende una matrice C e per ogni prodotto (colonna) ridireziona (rewire)
    una frazione eta dei suoi link in input verso altri prodotti scelti a caso.

    Input:
    - C: matrice di composizione (convenzione Cp'p: riga p' input di colonna p)
    - eta: frazione dei link in input da rewirare, per ogni prodotto (0 <= eta <= 1)
    - seed: opzionale, per riproducibilità

    Output:
    - C_rewired: nuova matrice con i link ridirezionati
    
    """
    # 0) GESTIONE 
    
    # 0.1) RANDOM SEED
    rng = np.random.default_rng(seed)
    
    # 0.2) VARIE
    C_mat = C.values.astype(float).copy()
    n = C_mat.shape[0]
    C_rewired = C_mat.copy()

    # 1) ALGORITMO 
    
    # iterazione sulle colonne
    for p in range(C_mat.shape[1]):
        colonna = C_rewired[:, p]

        # 1.a) indici di riga degli input attuali (righe p' con link = 1)
        input_attuali = np.where(colonna != 0)[0]
        n_link = len(input_attuali)

        # 1.b) numero di link da riwire
        n_rewire = int(np.floor(eta * n_link))
        if n_rewire == 0:
            continue

        # 1.c) scelta link da rimuovere
        da_rimuovere = rng.choice(input_attuali, size=n_rewire, replace=False) # rng choice (array, size = n, replace = False): estrae n elementi a caso da un array (False ---> senza rimpiazzo)


        # 1.d) scelta link da aggiungere
        
        # candidati per i nuovi link: righe che NON sono già input di p + p stesso
        gia_input = set(np.where(colonna != 0)[0])
        candidati = [i for i in range(n) if i not in gia_input and i != p]

        nuovi_input = rng.choice(candidati, size=n_rewire, replace=False)

        # 1.e) applicazione del rewiring
        C_rewired[da_rimuovere, p] = 0
        C_rewired[nuovi_input, p] = 1


    return pd.DataFrame(C_rewired, index=C.index, columns=C.columns)



def indegree_medio(C_df):
    """
    Calcola l'indegree medio della matrice C.
    
    """
    C = C_df.values
    
    return C.sum(axis=1).mean()




def nihilpotency(C: pd.DataFrame):
    """
    C e' nilpotente <=> il grafo diretto associato e' aciclico.
    Verificato via componenti fortemente connesse: aciclico <=> ogni SCC
    e' un singolo nodo e non ci sono self-loop.
    Costo O(V + E), esatto (nessun errore numerico).

    Va studiato bene come funziona questo algoritmo (funziona solo in casi specifici)

    Input:
    - C : matrice di composizione

    Output:
    - bool: mi dice se la matrice è nilpotente o meno
    
    """
    
    M = C.values.astype(int) 

    if np.any(np.diag(M) != 0):        # nel caso C contenga un self loop allora non è nilpotente 
        return False                   # questo non possiamo piu vederlo dopo

    A = csr_matrix(M == 1)  # mi interessa solo dove ci sono gli archi

    # numero di componenti fortemente connesse
    n_comp, _ = connected_components(A, directed=True, connection='strong') 

    # se ogni nodo è una componente fortemente connessa allora la matrice è nilpotente
    return n_comp == A.shape[0]



def compute_alpha(C: pd.DataFrame):
    """
    Calcola l'alpha da usare nell'algoritmo.
    - Se C e' nilpotente  -> rho = degree_medio(C)
    - Altrimenti          -> rho = autovettore_principale(C)
    Restituisce alpha = 1 / rho.
    """
    if nihilpotency(C):
        rho = indegree_medio(C)
    else:
        rho, _ = autovettore_principale(C)

    return 1.0 / rho



def flip_Mcp(Mcp: pd.DataFrame, eta, seed=None):
    """
    Flippa ogni entrata della matrice Mcp con probabilita' eta. 
    Stile Vito nel suo paper : "A New and Stable Estimation Method of Country
    Economic Fitness and Product Complexity"
    - Il flip è perciò simmetrico ---> se aumento eta aumenta anche la densità della matrice

    Input:
    - Mcp
    - eta : probabilità di un entrata di flippare
    - seed : per riproducibilità

    Output:
    - Mcp_flipped
    
    """
    M = Mcp.values.astype(int) 
    rng = np.random.default_rng(seed)
    # Indici della matrice da flippare
    flip = rng.random(M.shape) < eta # True se il valore random è minore di eta
    
    M_flipped = M.copy()
    M_flipped[flip] = 1.0 - M_flipped[flip] # =1 se Mcp = 0  | =0 se Mcp = 1
    
    return pd.DataFrame(M_flipped, index=Mcp.index, columns=Mcp.columns) 


def eades_order(C):
    """
    Da capire: per ora teniamoci il fatto che rende C nilpotente
    """
    
    A = (C == 1)
    n = A.shape[0]
    alive = np.ones(n, bool)
    indeg  = A.sum(1).astype(float)   # A[i,:] = input di i
    outdeg = A.sum(0).astype(float)   # A[:,i] = prodotti di cui i e' input
    s_head, s_tail = [], []
    while alive.any():
        moved = True
        while moved:
            moved = False
            for v in np.where(alive & (outdeg == 0))[0]:   # pozzi -> coda
                if alive[v]:
                    s_tail.append(v); alive[v] = False
                    indeg[A[:, v]] -= 1; outdeg[A[v, :]] -= 1; moved = True
            for v in np.where(alive & (indeg == 0))[0]:    # sorgenti -> testa
                if alive[v]:
                    s_head.append(v); alive[v] = False
                    indeg[A[:, v]] -= 1; outdeg[A[v, :]] -= 1; moved = True
        if alive.any():
            score = np.where(alive, outdeg - indeg, -np.inf)
            v = int(np.argmax(score))
            s_head.append(v); alive[v] = False
            indeg[A[:, v]] -= 1; outdeg[A[v, :]] -= 1
    return np.array(s_head + s_tail[::-1])


def make_C_nilpotente(C: pd.DataFrame):
    """
    Da capire: per ora teniamoci il fatto che rende C nilpotente
    """
    
    C_mat = C.values.astype(int)
    n = C.shape[0]
    order = eades_order(C_mat)
    pos = np.empty(n, int); pos[order] = np.arange(n)
    keep = pos[None, :] < pos[:, None]          # keep[i,j] = (pos[j] < pos[i])
    removed = (C_mat == 1) & (~keep)
    C_nilpotente = C_mat * keep
    C_nilpotente_df = pd.DataFrame(C_nilpotente, index=C.index, columns=C.columns) 
    return C_nilpotente_df

def compute_Q_medio_extended_fitness(
    Mcp: pd.DataFrame,
    C: pd.DataFrame,
    alpha: float,
    max_iter: int = 100000,
    tol: float = 1e-15,
    ):
    """
    history qui mi serve per riportare i valori di Q medio step by step
    
    """
    # 0) Gestione INPUT
    M  = Mcp.values.astype(float)
    C = C.values.astype(float)
    n_c, n_p = M.shape

    # 1) INIZIALIZZAZIONE
    F = np.ones(n_c)
    Q = np.ones(n_p)

    # *) SALVATAGGI OPZIONALI

    # *.a) Q MEDIO
    history    = [] 
   

    # 2) RUN DEL SISTEMA DINAMICO
    i = 0
    for i in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()

        #  2.1) EQUAZIONI SISTEMA DINAMICO
        F     = M @ Q_old

        Delta = 1.0 / (M.T @ (1.0 / F_old))
        Delta[np.isinf(Delta)] = 0.0
        
        mean_Delta = np.mean(Delta)
        Delta = Delta / mean_Delta

        CQ    = alpha * C @ Q_old
        Q     = CQ + Delta

        #  2.2) NORMALIZZAZIONE
        mean_Q = np.mean(Q)

        F     = F / np.mean(F)
        CQ    = CQ / mean_Q
        Delta = Delta / mean_Q
        Q     = Q / mean_Q
        
       
        history.append(mean_Q)
       

        # 3) CONVERGENZA
        if np.all(np.abs(F - F_old) < tol) and np.all(np.abs(Q - Q_old) < tol):
            break


    return mean_Q , history