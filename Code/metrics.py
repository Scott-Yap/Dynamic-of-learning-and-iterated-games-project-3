# metrics.py
"""
Metrics for Nash verification + Hart-style marginal comparison.

1) Exploitability / Nash gap for a pair (p,q):
      eps_A(p,q) = max_i (A q)_i  -  p^T A q
      eps_B(p,q) = p^T A q  -  min_j (p^T A)_j
      eps(p,q)   = max(eps_A, eps_B)

2) Random-battlefield marginal induced by p over allocations x in X:
      P_p(t) = (1/K) * sum_x p(x) * #{i : x_i = t},   t=0,...,S

3) Total variation distance:
      d_TV(P,Q) = 0.5 * sum_t |P(t) - Q(t)|
"""

import numpy as np

def value(p, q, A):
    """Game value under mixed strategies (p,q): v = p^T A q."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    A = np.asarray(A, dtype=float)
    return float(p @ (A @ q))


def exploitability(p, q, A):
    """
    Compute eps_A, eps_B, eps for (p,q) in a zero-sum matrix game with 
    payoff A for Player A.

    eps_A: how much Player A can gain by best-responding to q
    eps_B: how much Player B can gain by best-responding to p 

    Returns:
        eps, eps_A, eps_B, v
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    A = np.asarray(A, dtype=float)

    Aq = A @ q          # (n,) expected payoff of each pure row vs q
    pTA = p @ A         # (n,) expected payoff vs each pure column when mixing with p
    v = p @ Aq          # scalar game value under (p,q)

    eps_A = float(np.max(Aq) - v)
    eps_B = float(v - np.min(pTA))
    eps = max(eps_A, eps_B)

    return eps, eps_A, eps_B, v


def exploitability_over_time(p_hist, q_hist, A):
    """
    Compute exploitability per-iteration for sequences p_t, q_t.

    Inputs:
        p_hist: (T,n)
        q_hist: (T,n)
        A     : (n,n)

    Returns:
        eps_hist, epsA_hist, epsB_hist, v_hist   (each length T)
    """
    p_hist = np.asarray(p_hist, dtype=float)
    q_hist = np.asarray(q_hist, dtype=float)
    A = np.asarray(A, dtype=float)

    T = p_hist.shape[0]
    eps_hist = np.zeros(T)
    epsA_hist = np.zeros(T)
    epsB_hist = np.zeros(T)
    v_hist = np.zeros(T)

    for t in range(T):
        eps, epsA, epsB, v = exploitability(p_hist[t], q_hist[t], A)
        eps_hist[t] = eps
        epsA_hist[t] = epsA
        epsB_hist[t] = epsB
        v_hist[t] = v

    return eps_hist, epsA_hist, epsB_hist, v_hist


def running_average(hist):
    """
    Given hist (T,n), return avg_hist where avg_hist[t] = (1/(t+1)) sum_{s<=t} hist[s].
    """
    hist = np.asarray(hist, dtype=float)
    cumsum = np.cumsum(hist, axis=0)
    denom = (np.arange(hist.shape[0]) + 1).reshape(-1, 1)
    return cumsum / denom


def induced_random_battlefield_marginal(p, actions, S):
    """
    Compute the random-battlefield marginal P_p over troop counts.

    P_p(t) = (1/K) * sum_x p(x) * num{i : x_i = t},  for t=0,...,S

    Inputs:
        p       : length-n distribution over actions
        actions : list of tuples (x_1,...,x_K)
        S       : max troop count (if None, inferred as max over actions)

    Returns:
        P : length-(S+1) numpy array, sums to 1
    """
    p = np.asarray(p, dtype=float)
    n = len(actions)

    K = len(actions[0])

    P = np.zeros(S + 1, dtype=float)

    for prob, x in zip(p, actions):
        # each coordinate contributes prob/K to its count bin
        for v in x:
            P[v] += prob / K
    return P


def tv_distance(P, Q):
    """Total variation distance: 0.5 * sum |P - Q|."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return 0.5 * float(np.sum(np.abs(P - Q)))


def hart_target_marginal_B553():
    """
    Hart target marginal for B(5,5;3) as in your write-up:

      P(0)=P(2)=1/6
      P(1)=P(3)=1/3
      P(4)=P(5)=0
    """
    P = np.zeros(6, dtype=float)  # t=0..5
    P[0] = 1.0 / 6.0
    P[2] = 1.0 / 6.0
    P[1] = 1.0 / 3.0
    P[3] = 1.0 / 3.0
    # P[4]=P[5]=0
    return P


if __name__ == "__main__":
    # Test exploitability with rm.py + game.py + sym.py
    from game import make_game
    from rm import regret_matching
    from sym import symmetrise

    actions, idx, A = make_game(S=5, K=3)
    res = regret_matching(A, T=5000, seed=0)

    p_avg = res["p_avg"]
    print(p_avg)
    q_avg = res["q_avg"]

    eps, epsA, epsB, v = exploitability(p_avg, q_avg, A)
    print("exploitability:", eps, "(epsA:", epsA, "epsB:", epsB, ") value:", v)

    p_sym = symmetrise(p_avg, actions)
    q_sym = symmetrise(q_avg, actions)
    eps_sym, _, _, v_sym = exploitability(p_sym, q_sym, A)
    print("sym exploitability:", eps_sym, "sym value:", v_sym)

    P = induced_random_battlefield_marginal(p_sym, actions, S=5)
    Q = hart_target_marginal_B553()
    print("TV(P, Q):", tv_distance(P, Q))
