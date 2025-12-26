# run.py
"""
Run regret-matching Blotto experiments and save figures to ./figures/.

Figures (single main seed):
  - exploitability_vs_T.png
  - value_vs_T.png
  - tv_vs_T.png
  - marginal_bar.png

Figures (multi-seed, if enabled):
  - tv_vs_exploit_scatter.png
  - exploitability_rescaled_band.png
"""
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt

from game import make_game
from rm import regret_matching
from sym import build_perm_classes
from metrics import (
    exploitability,
    value,
    induced_random_battlefield_marginal,
    tv_distance,
    hart_target_marginal_B553,
)


def symmetrise_with_classes(p, perm_classes, n):
    """Project p onto permutation-invariant strategies by averaging within each class."""
    p = np.asarray(p, dtype=float)
    p_sym = np.zeros(n, dtype=float)
    for idxs in perm_classes.values():
        mass = p[idxs].sum()
        p_sym[idxs] = mass / len(idxs)
    return p_sym


def qstats(a):
    """Return (min, q25, median, q75, max)."""
    a = np.asarray(a, dtype=float)
    return (
        float(np.min(a)),
        float(np.quantile(a, 0.25)),
        float(np.median(a)),
        float(np.quantile(a, 0.75)),
        float(np.max(a)),
    )


def run_multiseed_scatter(A, actions, S, perm_classes, Q, T, seeds, outdir):
    """Scatter plot across seeds: TV-to-Hart vs exploitability (symmetrised averages)."""
    n = len(actions)
    eps_sym, tv = [], []

    for sd in seeds:
        res = regret_matching(A, T=T, seed=sd)
        p_avg, q_avg = res["p_avg"], res["q_avg"]

        p_sym = symmetrise_with_classes(p_avg, perm_classes, n)
        q_sym = symmetrise_with_classes(q_avg, perm_classes, n)

        eps_sym.append(float(exploitability(p_sym, q_sym, A)[0]))
        P = induced_random_battlefield_marginal(p_sym, actions, S=S)
        tv.append(float(tv_distance(P, Q)))

    eps_sym = np.array(eps_sym)
    tv = np.array(tv)

    plt.figure()
    plt.scatter(eps_sym, tv)
    plt.xlabel("Exploitability (symmetrised averages)")
    plt.ylabel("TV distance to Hart marginal")
    plt.title("TV distance to Hart across random seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tv_vs_exploit_scatter.png"), dpi=200)
    plt.close()

    return {"eps_sym": eps_sym, "tv": tv}


def multiseed_rescaled_band(A, actions, perm_classes, T, seeds, outdir, eval_every=50, burn_frac=0.2):
    """
    Median ± IQR band across seeds for rescaled convergence diagnostics:
      blue(t)   = sqrt(t) * eps(p̄_t, q̄_t)
      orange(t) = t * eps(p̄_t^sym, q̄_t^sym)
    """
    n = len(actions)
    seeds = list(seeds)

    eval_idx = np.arange(0, T, eval_every, dtype=int)
    x = eval_idx + 1  # time steps 1..T
    m = len(eval_idx)

    blue = np.zeros((len(seeds), m), dtype=float)
    orange = np.zeros((len(seeds), m), dtype=float)

    for s_i, sd in enumerate(seeds):
        res = regret_matching(A, T=T, seed=sd)
        p_hist, q_hist = res["p_hist"], res["q_hist"]

        p_cum = np.cumsum(p_hist, axis=0)
        q_cum = np.cumsum(q_hist, axis=0)

        for k, t_idx in enumerate(eval_idx):
            t = t_idx + 1
            pbar = p_cum[t_idx] / t
            qbar = q_cum[t_idx] / t

            eps = float(exploitability(pbar, qbar, A)[0])

            pbar_sym = symmetrise_with_classes(pbar, perm_classes, n)
            qbar_sym = symmetrise_with_classes(qbar, perm_classes, n)
            eps_sym = float(exploitability(pbar_sym, qbar_sym, A)[0])

            blue[s_i, k] = np.sqrt(t) * eps
            orange[s_i, k] = t * eps_sym

    def band(mat):
        med = np.median(mat, axis=0)
        q25 = np.quantile(mat, 0.25, axis=0)
        q75 = np.quantile(mat, 0.75, axis=0)
        return med, q25, q75

    blue_med, blue_q25, blue_q75 = band(blue)
    org_med, org_q25, org_q75 = band(orange)

    plt.figure()
    plt.plot(x, blue_med, label="median √T·eps (avg)")
    plt.fill_between(x, blue_q25, blue_q75, alpha=0.25)
    plt.plot(x, org_med, label="median T·eps (sym avg)")
    plt.fill_between(x, org_q25, org_q75, alpha=0.25)
    plt.title("Rescaled exploitability (median ± IQR across seeds)")
    plt.xlabel("T")
    plt.ylabel("rescaled exploitability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "exploitability_rescaled_band.png"), dpi=200)
    plt.close()

    burn = int(burn_frac * len(x))
    blue_tail = (float(np.median(blue_med[burn:])),
                 float(np.quantile(blue_med[burn:], 0.25)),
                 float(np.quantile(blue_med[burn:], 0.75)))
    org_tail = (float(np.median(org_med[burn:])),
                float(np.quantile(org_med[burn:], 0.25)),
                float(np.quantile(org_med[burn:], 0.75)))

    return {"blue_tail": blue_tail, "org_tail": org_tail}


def main(S=5, K=3, T=50000, outdir="figures", seed_main=0, seeds=None):
    os.makedirs(outdir, exist_ok=True)

    actions, idx, A = make_game(S=S, K=K)
    n = len(actions)

    perm_classes = build_perm_classes(actions)
    Q = hart_target_marginal_B553()

    # Main single-seed run (curves over time)
    res = regret_matching(A, T=T, seed=seed_main)
    p_hist, q_hist = res["p_hist"], res["q_hist"]

    x = np.arange(1, T + 1, dtype=int)
    p_cum = np.cumsum(p_hist, axis=0)
    q_cum = np.cumsum(q_hist, axis=0)

    eps_avg = np.zeros(T)
    eps_sym = np.zeros(T)
    vbar = np.zeros(T)
    tv_hist = np.zeros(T)

    for t_idx in range(T):
        t = t_idx + 1
        pbar = p_cum[t_idx] / t
        qbar = q_cum[t_idx] / t

        eps_avg[t_idx] = exploitability(pbar, qbar, A)[0]
        vbar[t_idx] = value(pbar, qbar, A)

        pbar_sym = symmetrise_with_classes(pbar, perm_classes, n)
        qbar_sym = symmetrise_with_classes(qbar, perm_classes, n)
        eps_sym[t_idx] = exploitability(pbar_sym, qbar_sym, A)[0]

        P = induced_random_battlefield_marginal(pbar_sym, actions, S=S)
        tv_hist[t_idx] = tv_distance(P, Q)

    # Plots: single-seed curves
    plt.figure()
    plt.plot(x, eps_avg, label="Exploitability (avg)")
    plt.plot(x, eps_sym, label="Exploitability (sym avg)")
    plt.xlabel("T")
    plt.ylabel("Exploitability")
    plt.title("Exploitability vs T")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "exploitability_vs_T.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, vbar, label="Value (avg strategies)", color="orange")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("T")
    plt.ylabel("Value")
    plt.title("Value vs T")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "value_vs_T.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, tv_hist)
    plt.xlabel("T")
    plt.ylabel("TV distance")
    plt.title("TV distance to Hart marginal vs T")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tv_vs_T.png"), dpi=200)
    plt.close()

    # Plot: marginal bar at final T
    pbar_T = p_cum[-1] / T
    pbar_T_sym = symmetrise_with_classes(pbar_T, perm_classes, n)
    P_learned = induced_random_battlefield_marginal(pbar_T_sym, actions, S=S)

    tgrid = np.arange(S + 1)
    width = 0.38

    plt.figure()
    plt.bar(tgrid - width / 2, P_learned, width=width, label="Learned marginal (sym)")
    plt.bar(tgrid + width / 2, Q,        width=width, label="Hart marginal")
    plt.xticks(tgrid)
    plt.xlabel("Troops on a random battlefield")
    plt.ylabel("Probability")
    plt.title("Learned marginal vs Hart target (final T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "marginal_bar.png"), dpi=200)
    plt.close()

    # Multi-seed diagnostics (optional)
    ms_stats = None
    band_stats = None
    if seeds:
        ms_stats = run_multiseed_scatter(A, actions, S, perm_classes, Q, T, seeds, outdir)
        band_stats = multiseed_rescaled_band(A, actions, perm_classes, T, seeds, outdir, eval_every=50)

    # Terminal output: report-ready summary only
    print(f"(S,K)=({S},{K}), |X|={n}, T={T}, seed_main={seed_main}")
    print("Main run (final T):")
    print(f"  exploitability(avg)       = {float(eps_avg[-1]):.6g}")
    print(f"  exploitability(sym avg)   = {float(eps_sym[-1]):.6g}")
    print(f"  value(avg strategies)     = {float(vbar[-1]):.6g}")
    print(f"  TV(sym marginal, Hart)    = {float(tv_hist[-1]):.6g}")

    if ms_stats is not None:
        eps_min, eps_q25, eps_med, eps_q75, eps_max = qstats(ms_stats["eps_sym"])
        tv_min, tv_q25, tv_med, tv_q75, tv_max = qstats(ms_stats["tv"])
        print(f"Across seeds (n={len(seeds)}), symmetrised averages:")
        print(f"  exploitability: median={eps_med:.6g}  IQR=[{eps_q25:.6g},{eps_q75:.6g}]  range=[{eps_min:.6g},{eps_max:.6g}]")
        print(f"  TV-to-Hart:     median={tv_med:.6g}   IQR=[{tv_q25:.6g},{tv_q75:.6g}]   range=[{tv_min:.6g},{tv_max:.6g}]")

    if band_stats is not None:
        blue_med, blue_q25, blue_q75 = band_stats["blue_tail"]
        org_med, org_q25, org_q75 = band_stats["org_tail"]
        print("Rescaled tail constants (median with IQR):")
        print(f"  √T·exploitability(avg):      median={blue_med:.3g}  IQR=[{blue_q25:.3g},{blue_q75:.3g}]")
        print(f"  T·exploitability(sym avg):   median={org_med:.3g}   IQR=[{org_q25:.3g},{org_q75:.3g}]")

    print(f"Saved figures to: {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--S", type=int, default=5)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--T", type=int, default=50000)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--seed_main", type=int, default=0)
    parser.add_argument("--nseeds", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="optional explicit list, e.g. --seeds 0 1 2 3")

    args = parser.parse_args()
    seeds = list(args.seeds) if args.seeds is not None else list(range(args.nseeds))

    main(S=args.S, K=args.K, T=args.T, outdir=args.outdir, seed_main=args.seed_main, seeds=seeds)

    # How to run:
    #   python run.py
    #   python run.py --T 5000
    #   python run.py --T 5000 --nseeds 50
    #   python run.py --T 5000 --seeds 0 1 2 3 4
