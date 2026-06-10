#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# In[6]:


def toy_C_1():
    """
    Matrice triangolare non nilpotente con tutte e 3 componenti fortemente connesse

    """
    return pd.DataFrame(
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]],
         #index=['A', 'B', 'C', 'D', 'E'],
        #columns=['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
    )




def toy_Mcp_1_A():
    """
    Matrice Mcp completamente nested

    """
    return pd.DataFrame(
         [[1, 1, 1],
         [1, 1, 0],
         [1, 0, 0]],
    )


def toy_Mcp_1_B():
    """
    Matrice Mcp non nested

    """
    return pd.DataFrame(
         [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
    )

















def generate_nested_M(n):
    M = np.zeros((n, n), dtype=int)
    
    for c in range(n):
        M[c, :n-c] = 1
        
    return pd.DataFrame(M)

def generate_C(n, edges):
    C = np.zeros((n, n), dtype=int)
    
    for i, j in edges:
        C[i, j] = 1
        
    return pd.DataFrame(C)

edges = [
    (2, 0),
    (3, 0),
    (3, 1),
    (4, 2),
    (4, 3),
    (5, 4)
]

C = generate_C(6, edges)

def generate_toy_model(n, edges):
    M = generate_nested_M(n)
    C = generate_C(n, edges)
    return M, C




def plot_C_network(C):
    G = nx.DiGraph()
    n = C.shape[0]

    G.add_nodes_from(range(n))

    # ATTENZIONE: direzione invertita
    for i in range(n):
        for j in range(n):
            if C.iloc[i, j] == 1:
                G.add_edge(j, i)  # ← inversione

    pos = nx.spring_layout(G, seed=42)

    labels = {i: i for i in range(n)}

    plt.figure(figsize=(6, 5))
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=800,
        arrows=True
    )

    plt.title("Product Network (C, transposed interpretation)")
    #plt.show()
# In[ ]:




