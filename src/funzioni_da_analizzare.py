#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def noise(compute_Q, Mcp, eta, n_rep=10, seed0=0):
    """RUMORE: quanto Q cambia sotto flip casuale di Mcp a intensita' eta."""
    M0 = Mcp.values.astype(float)
    Q0 = compute_Q(M0)
    d_sp, d_g = [], []
    for r in range(n_rep):
        Qp = compute_Q(flip_Mcp(M0, eta, seed=seed0 + r))
        m = (Q0 > 0) & (Qp > 0)
        d_sp.append(1 - spearmanr(Q0[m], Qp[m]).correlation)
        d_g.append(np.std(np.log(Qp[m]) - np.log(Q0[m])))
    return dict(sp=1 - np.mean(d_sp), one_minus_sp=np.mean(d_sp), gauge=np.mean(d_g))

def signal(compute_Q, Mcp_1, Mcp_2):
    """SEGNALE: quanto Q cambia fra due Mcp genuinamente diversi (due anni, o shock reale).
       Stessi prodotti (stesse colonne)."""
    Q1, Q2 = compute_Q(Mcp_1.values.astype(float)), compute_Q(Mcp_2.values.astype(float))
    m = (Q1 > 0) & (Q2 > 0)
    return dict(sp=spearmanr(Q1[m], Q2[m]).correlation,
                one_minus_sp=1 - spearmanr(Q1[m], Q2[m]).correlation,
                gauge=np.std(np.log(Q2[m]) - np.log(Q1[m])))

def signal_to_noise(compute_Q, Mcp_1, Mcp_2, eta=0.1, n_rep=10):
    s = signal(compute_Q, Mcp_1, Mcp_2)
    n = noise(compute_Q, Mcp_1, eta=eta, n_rep=n_rep)
    return dict(signal=s['one_minus_sp'], noise=n['one_minus_sp'],
                ratio=s['one_minus_sp'] / n['one_minus_sp'])
