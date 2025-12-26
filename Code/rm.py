# rm.py
"""
Regret Matching (sampled update) for Colonel Blotto in the style of Neller & Lanctot.

At each time t:
  p_t = normalize((R^A)^+)  or uniform if all nonpositive
  q_t = normalize((R^B)^+)  or uniform if all nonpositive

Then sample pure actions:
  i ~ p_t,  j ~ q_t
  u_t = A[i, j]   (realised payoff to A). B gets -u_t.

Regret updates (external regret, realised):
  R^A += A[:, j] - u_t
  R^B += u_t - A[i, :]

Diagnostics:
  v_t = p_t^T A q_t  (expected payoff under mixed strategies)

Averages:
  p_avg = (1/T) sum_t p_t
  q_avg = (1/T) sum_t q_t
"""
import numpy as np


def positive_part(x):
    """(x)^+ elementwise."""
    return np.maximum(x, 0.0)


def strategy_from_regret(R):
    """
    Convert a regret vector R into a mixed strategy.
    If all entries are nonpositive, return uniform strategy.
    """
    Rp = positive_part(R)
    s = Rp.sum()
    n = R.shape[0]
    if s > 0.0:
        return Rp / s
    return np.ones(n) / n


def regret_matching(A, T, seed=None):
    """
    Run sampled RM self-play for T iterations (Neller–Lanctot style).

    Returns a dict containing:
      p_avg, q_avg    : time-averaged strategies (averaging the mixed strategies p_t, q_t)
      u_hist          : realised per-iteration payoffs u_t = A[i_t, j_t]
      u_avg_hist      : running average of realised payoffs
      v_hist          : expected per-iteration payoffs v_t = p_t^T A q_t
      v_avg_hist      : running average of expected payoffs
      p_hist, q_hist  : per-iteration mixed strategies
      i_hist, j_hist  : sampled pure actions
      RA, RB          : final regret vectors
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n)

    rng = np.random.default_rng(seed)

    RA = np.zeros(n)
    RB = np.zeros(n)

    p_sum = np.zeros(n)
    q_sum = np.zeros(n)

    u_hist = np.zeros(T)
    u_avg_hist = np.zeros(T)

    v_hist = np.zeros(T)
    v_avg_hist = np.zeros(T)

    p_hist = np.zeros((T, n))
    q_hist = np.zeros((T, n))

    i_hist = np.zeros(T, dtype=int)
    j_hist = np.zeros(T, dtype=int)

    running_u = 0.0
    running_v = 0.0

    for t in range(T):
        # 1) Current mixed strategies from positive regrets
        p = strategy_from_regret(RA)
        q = strategy_from_regret(RB)

        # 2) Expected payoff under mixed strategies (diagnostic)
        v = p @ (A @ q)

        # 3) Sample realised pure actions
        i = rng.choice(n, p=p)  # A's realised action index
        j = rng.choice(n, p=q)  # B's realised action index
        u = A[i, j]             # realised payoff to A

        # 4) Realised external-regret update
        RA += A[:, j] - u
        RB += u - A[i, :]

        # 5) Accumulate averages of the mixed strategies
        p_sum += p
        q_sum += q

        # 6) Logging
        u_hist[t] = u
        running_u += u
        u_avg_hist[t] = running_u / (t + 1)

        v_hist[t] = v
        running_v += v
        v_avg_hist[t] = running_v / (t + 1)

        p_hist[t] = p
        q_hist[t] = q

        i_hist[t] = i
        j_hist[t] = j

    p_avg = p_sum / T
    q_avg = q_sum / T

    return {
        "p_avg": p_avg,
        "q_avg": q_avg,
        "u_hist": u_hist,
        "u_avg_hist": u_avg_hist,
        "v_hist": v_hist,
        "v_avg_hist": v_avg_hist,
        "RA": RA,
        "RB": RB,
        "p_hist": p_hist,
        "q_hist": q_hist,
        "i_hist": i_hist,
        "j_hist": j_hist,
        "v_final_avg_strats": p_avg @ A @ q_avg,
    }


if __name__ == "__main__":
    from game import make_game

    actions, idx, A = make_game(S=5, K=3)
    res = regret_matching(A, T=5000, seed=0)

    print("final running avg realised payoff (ū_T):", res["u_avg_hist"][-1])
    print("final running avg expected payoff  (v̄_T):", res["v_avg_hist"][-1])
    print("expected payoff of averaged strats p̄^T A q̄:", 
          res["v_final_avg_strats"])

    print("p_avg:", res["p_avg"])
    print("q_avg:", res["q_avg"])
    print("sum p_avg:", res["p_avg"].sum())
    print("sum q_avg:", res["q_avg"].sum())
