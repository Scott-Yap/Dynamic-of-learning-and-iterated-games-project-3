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
  - marginal_bar_best_tv.png
"""
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")  # set backend before importing pyplot
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


def save_fig(outdir, filename, dpi=200):
    """Save the current matplotlib figure and close it."""
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=dpi)
    plt.close()


def run_multiseed_scatter(A, actions, S, perm_classes, Q, T, seeds, outdir):
    """Scatter across seeds: TV-to-Hart vs exploitability (symmetrised averages)."""
    n = len(actions)
    eps_avg, eps_sym, tv = [], [], []
    P_list = []  # learned marginals per seed (symmetrised)

    for sd in seeds:
        res = regret_matching(A, T=T, seed=sd)
        p_avg, q_avg = res["p_avg"], res["q_avg"]

        # exploitability of raw averages
        eps_avg.append(float(exploitability(p_avg, q_avg, A)[0]))

        # symmetrised averages (for Hart comparison)
        p_sym = symmetrise_with_classes(p_avg, perm_classes, n)
        q_sym = symmetrise_with_classes(q_avg, perm_classes, n)
        eps_sym.append(float(exploitability(p_sym, q_sym, A)[0]))

        # induced marginal and TV distance to Hart target marginal
        P = induced_random_battlefield_marginal(p_sym, actions, S=S)
        P_list.append(P)
        tv.append(float(tv_distance(P, Q)))

    eps_avg = np.array(eps_avg)
    eps_sym = np.array(eps_sym)
    tv = np.array(tv)

    plt.figure()
    plt.scatter(eps_sym, tv)
    plt.xlabel("Exploitability (symmetrised averages)")
    plt.ylabel("TV distance to Hart marginal")
    plt.title("TV distance to Hart across random seeds")
    save_fig(outdir, "tv_vs_exploit_scatter.png")

    return {
        "eps_avg": eps_avg,
        "eps_sym": eps_sym,
        "tv": tv,
        "P_list": P_list,
        "seeds": list(seeds),
    }


def multiseed_rescaled_band(A, actions, perm_classes, T, seeds, outdir, eval_every=50, burn_frac=0.2):
    """
    Median and IQR band across seeds for rescaled diagnostics:
      blue_s(t)   = sqrt(t) * exploitability(pbar_t, qbar_t)
      orange_s(t) = t * exploitability(pbar_t_sym, qbar_t_sym)

    Plot:
      solid = median across seeds at each evaluation time
      band  = IQR across seeds at each evaluation time

    Summary:
      tail constants are computed per seed as the median over late evaluation times,
      then reported as median and IQR across seeds.
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

        # cumulative sums let us form pbar_t and qbar_t quickly
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
    plt.plot(x, blue_med, label="median sqrt(T)*Exploit (avg)")
    plt.fill_between(x, blue_q25, blue_q75, alpha=0.25)
    plt.plot(x, org_med, label="median T*Exploit (sym avg)")
    plt.fill_between(x, org_q25, org_q75, alpha=0.25)
    plt.title("Rescaled exploitability (median and IQR across seeds)")
    plt.xlabel("T")
    plt.ylabel("rescaled exploitability")
    plt.legend()
    save_fig(outdir, "exploitability_rescaled_band.png")

    burn_k = int(burn_frac * m)

    # per-seed tail constants: median over late evaluation times
    blue_const_per_seed = np.median(blue[:, burn_k:], axis=1)
    org_const_per_seed = np.median(orange[:, burn_k:], axis=1)

    def seed_stats(v):
        return (
            float(np.median(v)),
            float(np.quantile(v, 0.25)),
            float(np.quantile(v, 0.75)),
        )

    blue_tail = seed_stats(blue_const_per_seed)
    org_tail = seed_stats(org_const_per_seed)

    # across-seed band at final evaluation time
    blue_final = (float(blue_med[-1]), float(blue_q25[-1]), float(blue_q75[-1]))
    org_final = (float(org_med[-1]), float(org_q25[-1]), float(org_q75[-1]))

    return {
        "blue_tail": blue_tail,
        "org_tail": org_tail,
        "blue_final": blue_final,
        "org_final": org_final,
    }


def plot_single_seed_curves(outdir, x, eps_avg, eps_sym, vbar, vbar_sym, tv_hist):
    """Save the single-seed diagnostic curves."""
    plt.figure()
    plt.plot(x, eps_avg, label="Exploitability (avg)")
    plt.plot(x, eps_sym, label="Exploitability (sym avg)")
    plt.xlabel("T")
    plt.ylabel("Exploitability")
    plt.title("Exploitability vs T")
    plt.legend()
    save_fig(outdir, "exploitability_vs_T.png")

    plt.figure()
    plt.plot(x, vbar, label="Value (avg strategies)")
    plt.plot(x, vbar_sym, label="Value (sym avg strategies)")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("T")
    plt.ylabel("Value")
    plt.title("Value vs T")
    plt.legend()
    save_fig(outdir, "value_vs_T.png")

    plt.figure()
    plt.plot(x, tv_hist)
    plt.xlabel("T")
    plt.ylabel("TV distance")
    plt.title("TV distance to Hart marginal vs T")
    save_fig(outdir, "tv_vs_T.png")


def plot_marginal_bar(outdir, S, P_learned, Q, filename, title):
    """Save a side-by-side bar plot for learned marginal vs Hart target marginal."""
    tgrid = np.arange(S + 1)
    width = 0.38

    plt.figure()
    plt.bar(tgrid - width / 2, P_learned, width=width, label="Learned marginal (sym)")
    plt.bar(tgrid + width / 2, Q, width=width, label="Hart marginal")
    plt.xticks(tgrid)
    plt.xlabel("Troops on a random battlefield")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    save_fig(outdir, filename)


def main(S=5, K=3, T=50000, outdir="figures", seed_main=0, seeds=None):
    os.makedirs(outdir, exist_ok=True)

    # Build the game in matrix form
    actions, idx, A = make_game(S=S, K=K)
    n = len(actions)

    # Permutation classes and Hart target marginal for (S,K)=(5,3)
    perm_classes = build_perm_classes(actions)
    Q = hart_target_marginal_B553()

    # Single-seed run to produce curves over time
    res = regret_matching(A, T=T, seed=seed_main)
    p_hist, q_hist = res["p_hist"], res["q_hist"]

    x = np.arange(1, T + 1, dtype=int)
    p_cum = np.cumsum(p_hist, axis=0)
    q_cum = np.cumsum(q_hist, axis=0)

    eps_avg = np.zeros(T, dtype=float)
    eps_sym = np.zeros(T, dtype=float)
    vbar = np.zeros(T, dtype=float)
    vbar_sym = np.zeros(T, dtype=float)
    tv_hist = np.zeros(T, dtype=float)

    # Evaluate diagnostics for each time t using time-averaged strategies
    for t_idx in range(T):
        t = t_idx + 1
        pbar = p_cum[t_idx] / t
        qbar = q_cum[t_idx] / t

        eps_avg[t_idx] = float(exploitability(pbar, qbar, A)[0])
        vbar[t_idx] = float(value(pbar, qbar, A))

        pbar_sym = symmetrise_with_classes(pbar, perm_classes, n)
        qbar_sym = symmetrise_with_classes(qbar, perm_classes, n)

        eps_sym[t_idx] = float(exploitability(pbar_sym, qbar_sym, A)[0])
        vbar_sym[t_idx] = float(value(pbar_sym, qbar_sym, A))

        P = induced_random_battlefield_marginal(pbar_sym, actions, S=S)
        tv_hist[t_idx] = float(tv_distance(P, Q))

    # Save single-seed figures
    plot_single_seed_curves(outdir, x, eps_avg, eps_sym, vbar, vbar_sym, tv_hist)

    # Save marginal comparison at final time T
    pbar_T = p_cum[-1] / T
    pbar_T_sym = symmetrise_with_classes(pbar_T, perm_classes, n)
    P_learned = induced_random_battlefield_marginal(pbar_T_sym, actions, S=S)
    plot_marginal_bar(
        outdir=outdir,
        S=S,
        P_learned=P_learned,
        Q=Q,
        filename="marginal_bar.png",
        title="Learned marginal vs Hart target (final T)",
    )

    # Multi-seed diagnostics (optional, enabled when seeds is non-empty)
    ms_stats = None
    band_stats = None
    if seeds:
        ms_stats = run_multiseed_scatter(A, actions, S, perm_classes, Q, T, seeds, outdir)
        band_stats = multiseed_rescaled_band(A, actions, perm_classes, T, seeds, outdir, eval_every=50)

        # Best seed by TV distance: save an additional marginal plot
        best_i = int(np.argmin(ms_stats["tv"]))
        best_seed = ms_stats["seeds"][best_i]
        best_tv = float(ms_stats["tv"][best_i])
        best_P = ms_stats["P_list"][best_i]

        tgrid = np.arange(S + 1)
        width = 0.38
        plt.figure()
        plt.bar(
            tgrid - width / 2,
            best_P,
            width=width,
            label=f"Learned marginal (sym), seed={best_seed}",
        )
        plt.bar(tgrid + width / 2, Q, width=width, label="Hart marginal")
        plt.xticks(tgrid)
        plt.xlabel("Troops on a random battlefield")
        plt.ylabel("Probability")
        plt.title(f"Learned marginal vs Hart (best TV seed, TV={best_tv:.4g})")
        plt.legend()
        save_fig(outdir, "marginal_bar_best_tv.png")

    # Terminal output: report-ready summary only
    print(f"(S,K)=({S},{K}), |X|={n}, T={T}, seed_main={seed_main}")
    print("Main run (final T):")
    print(f"  exploitability(avg)       = {float(eps_avg[-1]):.6g}")
    print(f"  exploitability(sym avg)   = {float(eps_sym[-1]):.6g}")
    print(f"  value(avg strategies)     = {float(vbar[-1]):.6g}")
    print(f"  value(sym avg strategies) = {float(vbar_sym[-1]):.6g}")
    print(f"  TV(sym marginal, Hart)    = {float(tv_hist[-1]):.6g}")

    if ms_stats is not None:
        avg_min, avg_q25, avg_med, avg_q75, avg_max = qstats(ms_stats["eps_avg"])
        sym_min, sym_q25, sym_med, sym_q75, sym_max = qstats(ms_stats["eps_sym"])
        tv_min, tv_q25, tv_med, tv_q75, tv_max = qstats(ms_stats["tv"])

        print(f"Across seeds (n={len(seeds)}):")
        print(
            "  exploitability(avg):     "
            f"median={avg_med:.6g}  IQR=[{avg_q25:.6g},{avg_q75:.6g}]  range=[{avg_min:.6g},{avg_max:.6g}]"
        )
        print(
            "  exploitability(sym avg): "
            f"median={sym_med:.6g}  IQR=[{sym_q25:.6g},{sym_q75:.6g}]  range=[{sym_min:.6g},{sym_max:.6g}]"
        )
        print(
            "  TV-to-Hart (sym):        "
            f"median={tv_med:.6g}   IQR=[{tv_q25:.6g},{tv_q75:.6g}]   range=[{tv_min:.6g},{tv_max:.6g}]"
        )

        best_i = int(np.argmin(ms_stats["tv"]))
        print(
            f"  best-TV seed: {ms_stats['seeds'][best_i]}  TV={float(ms_stats['tv'][best_i]):.6g}  "
            f"exploit(sym)={float(ms_stats['eps_sym'][best_i]):.6g}"
        )
        print(f"  saved: {outdir}/marginal_bar_best_tv.png")

    if band_stats is not None:
        blue_med, blue_q25, blue_q75 = band_stats["blue_tail"]
        org_med, org_q25, org_q75 = band_stats["org_tail"]
        print("Rescaled tail constants (median with IQR across seeds):")
        print(f"  sqrt(T)*exploitability(avg):    median={blue_med:.3g}  IQR=[{blue_q25:.3g},{blue_q75:.3g}]")
        print(f"  T*exploitability(sym avg):      median={org_med:.3g}  IQR=[{org_q25:.3g},{org_q75:.3g}]")

        b_med, b_q25, b_q75 = band_stats["blue_final"]
        o_med, o_q25, o_q75 = band_stats["org_final"]
        print("Rescaled values at final T (median with IQR across seeds):")
        print(f"  sqrt(T)*exploitability(avg):    median={b_med:.3g}  IQR=[{b_q25:.3g},{b_q75:.3g}]")
        print(f"  T*exploitability(sym avg):      median={o_med:.3g}  IQR=[{o_q25:.3g},{o_q75:.3g}]")

    print(f"Saved figures to: {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--S", type=int, default=5)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--T", type=int, default=50000)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--seed_main", type=int, default=0)
    parser.add_argument("--nseeds", type=int, default=20)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="optional explicit list, e.g. --seeds 0 1 2 3",
    )

    args = parser.parse_args()
    seeds = list(args.seeds) if args.seeds is not None else list(range(args.nseeds))

    main(S=args.S, K=args.K, T=args.T, outdir=args.outdir, seed_main=args.seed_main, seeds=seeds)

    # How to run:
    #   python run.py
    #   python run.py --T 10000
    #   python run.py --T 10000 --nseeds 50 (used in the project)
    #   python run.py --T 10000 --seeds 0 1 2 3 4