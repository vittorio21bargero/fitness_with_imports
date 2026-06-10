#!/usr/bin/env python
# coding: utf-8

import numpy as np

# In[ ]:


def phi_1(M: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    F_c = sum_p M_cp * Q_p
    """
    return M @ Q

def phi_2(M_E, Q, penalty):
    """
    Fitness con penalità sugli import.
    F_c = sum_p( M_E[c,p] * Q[p] ) - penalty_c
    
    """
    raw = M_E @ Q - penalty          # shape (n_c,)
    mean = raw.mean()
    return raw / mean if mean != 0 else raw


# In[ ]:


def theta_1(F: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Q_p = 1 / sum_c (M_cp * (1 / F_c))
    """
    return 1.0 / (M.T @ (1.0 / F))


def matrix_power_series(C: np.ndarray, max_power: int):
    """
    Calcola S = I + C + C^2 + ... + C^k

    Parametri:
    ----------
    C : matrice C di composizione  (n x n)
    max_power : potenza massima a cui voglio far arrivare la sommatoria

    Returns
    -------
    S : np.array  matrice con la somma delle potenze
    
    """

    # dimensione matrice C
    n = C.shape[0]
    # inizializzo le matrici S con la stessa dimensione di C chiaramente
    S = np.eye(n)
    current_power = np.eye(n)

    for _ in range(1, max_power + 1):
        current_power = current_power @ C
        S += current_power

    return S
def build_C_tensor(C, M):
    P = C.shape[0]
    C_tensor = np.zeros((P, P, M))
    C_k = C.copy()
    for k in range(M):
        C_tensor[:, :, k] = C_k
        C_k = C_k @ C
    return C_tensor


def build_filter_tensors(M_I, M_E, M):
    M_I_tensor = np.repeat(M_I[:, :, np.newaxis], M, axis=2)
    M_E_tensor = np.repeat(M_E[:, :, np.newaxis], M, axis=2)
    return M_I_tensor, M_E_tensor


def compute_penalty(Delta, M_I_tensor, M_E_tensor, C_tensor):
    """
    FIX: M_I_tensor e M_E_tensor ora sono parametri espliciti.
    FIX: rinominato filtered → filtered_1, typo filterd_2 → filtered_2.
    """
    Cn, P, M = Delta.shape
    penalty = np.zeros(Cn)

    for k in range(M):
        # FILTRO 1: import  (shape: Cn x P)
        filtered_1 = Delta[:, :, k] * M_I_tensor[:, :, k]

        # Propagazione lungo filiera  (shape: Cn x P)
        propagated = filtered_1 @ C_tensor[:, :, k]

        # FILTRO 2: export  →  somma su prodotti  (shape: Cn)
        filtered_2 = (propagated * M_E_tensor[:, :, k]).sum(axis=1)
        penalty += filtered_2

    return penalty


def shift_and_scale_delta(Delta, Delta_new, mean_Q):
    Delta_updated = np.empty_like(Delta)
    Delta_updated[:, :, 1:] = Delta[:, :, :-1] / mean_Q
    Delta_updated[:, :, 0] = Delta_new / mean_Q
    return Delta_updated


def normalize(x):
    s = x.mean()
    return x / s if s != 0 else x

