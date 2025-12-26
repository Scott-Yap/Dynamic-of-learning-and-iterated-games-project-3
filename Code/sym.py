# sym.py
"""
Symmetrisation layer over battlefield permutations.

We group actions into permutation-classes ("perm-classes") by sorted 
coordinates.

Given a distribution p over pure allocations (indexed like `actions`),
Sym(p) is defined by:
  - perm-class representative: c = sort(x)
  - perm-class mass: m_c = sum_{x: sort(x)=c} p(x)
  - redistribute uniformly within the perm-class:
      Sym(p)(x) = m_c / |perm(c)|  for all x with sort(x)=c

Also includes an optional "perm-class utility" reducer:
  U[c,d] = average payoff between perm-class c and perm-class d
         = (1/|perm(c)||perm(d)|) sum_{i in perm(c)} sum_{j in perm(d)} A[i,j]
"""

from collections import defaultdict
import numpy as np


def perm_rep(x):
    """Permutation-class representative: sorted coordinates."""
    return tuple(sorted(x))


def build_perm_classes(actions):
    """
    Partition actions into permutation-classes by sorted coordinates.

    Returns:
        perm_classes: dict rep -> list of indices in `actions` belonging to 
        that perm-class
    """
    perm_classes = defaultdict(list)
    for i, x in enumerate(actions):
        perm_classes[perm_rep(x)].append(i)
    return dict(perm_classes)


def symmetrise(p, actions):
    """
    Compute Sym(p) as a vector aligned with `actions`.

    Inputs:
        p           : length-n vector (distribution over actions, 
        should sum to 1)
        actions     : list of pure allocations (tuples)
        perm_classes: optional precomputed build_perm_classes(actions)

    Returns:
        p_sym       : length-n vector, symmetrised distribution
    """
    p = np.asarray(p, dtype=float)
    n = len(actions)

    perm_classes = build_perm_classes(actions)

    p_sym = np.zeros(n, dtype=float)

    # For each perm-class: compute mass, spread it uniformly within the class.
    for rep, idxs in perm_classes.items():
        mass = p[idxs].sum()
        p_sym[idxs] = mass / len(idxs)

    return p_sym


if __name__ == "__main__":
    # quick smoke test if you have game.py + rm.py
    from game import make_game
    from rm import regret_matching

    actions, idx, A = make_game(S=5, K=3)
    res = regret_matching(A, T=5000, seed=0)

    p_sym = symmetrise(res["p_avg"], actions)
    q_sym = symmetrise(res["q_avg"], actions)

    print(len(p_sym))
    print(p_sym)
    print("sum p_avg:", res["p_avg"].sum(), "sum p_sym:", p_sym.sum())