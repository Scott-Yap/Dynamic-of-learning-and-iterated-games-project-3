# game.py

"""
Minimal utilities for the (S,K) Colonel Blotto instance:
1) enumerate all pure allocations x in Z_{\ge 0}^K with sum x_i = S
2) build the payoff matrix A where A[i,j] = u(x^(i), x^(j))

Payoff:
  u(x,y) = (1/K) * sum_{k=1}^K sgn(x_k - y_k)    in [-1, 1]
"""
import numpy as np


def enumerate_actions(S, K):
    """
    Return a list of all K-tuples of nonnegative integers summing to S.
    Example (S=5,K=3): (0,0,5), (0,1,4), ..., (5,0,0)
    """
    actions = []

    def rec(prefix, remaining, slots_left):
        # Fill remaining slots so total sum is S
        if slots_left == 1:
            actions.append(tuple(prefix + [remaining]))
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, slots_left - 1)

    rec([], S, K)

    return actions


def sgn(z):
    """Sign function returning -1, 0, +1."""
    return (z > 0) - (z < 0)


def u(x, y):
    """
    Payoff:
        u(x,y) = (1/K) * sum_k sgn(x_k - y_k)
    """
    K = len(x)
    return sum(sgn(xk - yk) for xk, yk in zip(x, y)) / K


def build_payoff_matrix(actions):
    """
    Build A where A[i,j] = u(actions[i], actions[j]).
    """
    n = len(actions)
    A = np.zeros((n, n), dtype=float)

    for i, xi in enumerate(actions):
        for j, yj in enumerate(actions):
            A[i, j] = u(xi, yj)

    return A


def make_game(S, K):
    """
    Convenience function to build everything for an instance (S,K).

    Returns:
        actions : list of pure allocations (tuples)
        idx     : dict mapping action tuple -> row/col index in A
        A       : payoff matrix, A[i,j] = u(actions[i], actions[j])
    """
    actions = enumerate_actions(S, K)
    idx = {a: i for i, a in enumerate(actions)}
    A = build_payoff_matrix(actions)
    return actions, idx, A


if __name__ == "__main__":
    # Running instance: (S,K) = (5,3)
    actions, idx, A = make_game(S=5, K=3)
    print(idx.keys())
    print("|X| =", len(actions))
    print("A shape =", A.shape)
    print("first 5 actions:", actions[:5])
