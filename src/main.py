"""
End-to-end pipeline
====================
1. Runs the 100k-agent ABM to generate simulated returns for 50 stocks / 5 sectors.
2. Loads real (or synthetic fallback) S&P 500 returns for 40+ stocks / 5 sectors.
3. Builds correlation networks for both and detects communities.
4. Computes RMT (Marchenko-Pastur) eigenvalue spectra for both and compares them.
5. Saves all plots + a text summary to results/.

Usage:
    python src/main.py                      # full run, defaults
    python src/main.py --n-agents 20000 --n-steps 250   # faster smoke test
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))

from abm.model import MarketABM, ABMConfig
from analysis.data import fetch_sp500_returns, ticker_sector_map, SP500_TICKERS
from analysis.network import (
    correlation_matrix,
    build_correlation_graph,
    detect_communities,
    community_sector_agreement,
)
from analysis.rmt import (
    eigen_spectrum,
    marchenko_pastur_pdf,
    marchenko_pastur_bounds,
    signal_vs_noise_eigenvalues,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def parse_args():
    p = argparse.ArgumentParser(description="Segregation-of-Financial-Markets ABM pipeline")
    p.add_argument("--n-agents", type=int, default=100_000)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--n-sectors", type=int, default=5)
    p.add_argument("--stocks-per-sector", type=int, default=10)
    p.add_argument("--corr-threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh-data", action="store_true", help="force re-download of real market data")
    return p.parse_args()


def run_abm(args) -> MarketABM:
    print(f"\n[1/4] Running ABM: {args.n_agents:,} agents, {args.n_steps} steps, "
          f"{args.n_sectors} sectors x {args.stocks_per_sector} stocks...")
    cfg = ABMConfig(
        n_agents=args.n_agents,
        n_sectors=args.n_sectors,
        stocks_per_sector=args.stocks_per_sector,
        seed=args.seed,
    )
    model = MarketABM(cfg)
    model.run(n_steps=args.n_steps, verbose=True)
    return model


def plot_abm_diagnostics(model: MarketABM):
    R = model.returns_matrix()
    seg = model.segregation_series()
    regime = model.regime_series()

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(seg, color="#2c7fb8")
    axes[0].set_ylabel("Segregation index")
    axes[0].set_title("Emergent segregation of trading agents across stock communities")

    market_ret = R.mean(axis=1)
    axes[1].plot(market_ret, color="#41ab5d", lw=0.8)
    axes[1].set_ylabel("Mean simulated return\n(avg across 50 stocks)")
    axes[1].set_title("Simulated 'market' return series")

    axes[2].fill_between(np.arange(len(regime)), regime, step="mid", color="#e34a33", alpha=0.6)
    axes[2].set_ylabel("Crisis regime")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["calm", "crisis"])
    axes[2].set_xlabel("Simulation step")
    axes[2].set_title("Volatility regime (Markov-switching herding/vol)")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "abm_diagnostics.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(corr, labels, title, filename):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_network(G, sector_of, title, filename, seed=7):
    fig, ax = plt.subplots(figsize=(9, 9))
    pos = nx.spring_layout(G, seed=seed, k=0.6 / max(1, np.sqrt(len(G))))

    sectors = sorted(set(sector_of.values()))
    cmap = plt.get_cmap("tab10")
    color_of_sector = {s: cmap(i) for i, s in enumerate(sectors)}
    node_colors = [color_of_sector.get(sector_of.get(n), "grey") for n in G.nodes()]

    weights = [abs(G[u][v].get("weight", 0.3)) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=[2 * w for w in weights])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=220, edgecolors="black", linewidths=0.4)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_of_sector[s],
                           markersize=9, label=s) for s in sectors]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_eigenvalue_comparison(real_eigs, sim_eigs, n_assets_real, n_obs_real,
                                n_assets_sim, n_obs_sim):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, eigs, n_assets, n_obs, label in [
        (axes[0], real_eigs, n_assets_real, n_obs_real, "Real (or real-like) S&P 500 stocks"),
        (axes[1], sim_eigs, n_assets_sim, n_obs_sim, "ABM-simulated stocks"),
    ]:
        ax.hist(eigs, bins=30, density=True, alpha=0.6, color="#3182bd", label="empirical eigenvalues")
        x = np.linspace(max(1e-6, eigs.min() * 0.5), eigs.max() * 1.05, 400)
        mp_pdf = marchenko_pastur_pdf(x, n_assets, n_obs)
        ax.plot(x, mp_pdf, color="#e34a33", lw=2, label="Marchenko-Pastur (noise) prediction")
        lmin, lmax = marchenko_pastur_bounds(n_assets, n_obs)
        ax.axvline(lmax, color="black", ls="--", lw=1, label=f"MP upper edge = {lmax:.2f}")
        ax.set_title(label)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.suptitle("Eigenvalue spectrum vs Random Matrix Theory (Marchenko-Pastur) null model")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eigenvalue_rmt_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------- 1. ABM ----------------
    model = run_abm(args)
    plot_abm_diagnostics(model)
    sim_returns = model.returns_matrix()
    sim_labels = model.stock_labels()
    sim_sector_of = {lbl: f"Sector{model.stock_sector[i]}" for i, lbl in enumerate(sim_labels)}

    print("[1/4] Done. Segregation index converged to "
          f"{model.segregation_series()[-1]:.3f}. "
          f"Fraction of steps in crisis regime: {model.regime_series().mean():.1%}")

    # ---------------- 2. Real market data ----------------
    print("\n[2/4] Loading real S&P 500 data (falls back to synthetic if offline)...")
    real_returns_df, used_real = fetch_sp500_returns(force_refresh=args.refresh_data)
    real_sector_of = ticker_sector_map()
    print(f"[2/4] Loaded {real_returns_df.shape[1]} stocks x {real_returns_df.shape[0]} "
          f"trading days. Using real market data: {used_real}")

    # ---------------- 3. Correlation networks ----------------
    print("\n[3/4] Building correlation networks + detecting communities...")
    real_corr = correlation_matrix(real_returns_df)
    real_graph = build_correlation_graph(real_corr, threshold=args.corr_threshold)
    real_partition = detect_communities(real_graph)
    real_purity = community_sector_agreement(real_partition, real_sector_of)
    plot_correlation_heatmap(real_corr.values, real_corr.columns.tolist(),
                              "Real (or real-like) S&P 500 correlation matrix",
                              "real_correlation_heatmap.png")
    plot_network(real_graph, real_sector_of,
                 f"Real market correlation network (|r| >= {args.corr_threshold}), "
                 f"community purity = {real_purity:.2f}",
                 "real_correlation_network.png")

    sim_corr_df = None
    import pandas as pd
    sim_returns_df = pd.DataFrame(sim_returns, columns=sim_labels)
    sim_corr = correlation_matrix(sim_returns_df)
    sim_graph = build_correlation_graph(sim_corr, threshold=args.corr_threshold)
    sim_partition = detect_communities(sim_graph)
    sim_purity = community_sector_agreement(sim_partition, sim_sector_of)
    plot_correlation_heatmap(sim_corr.values, sim_corr.columns.tolist(),
                              "ABM-simulated stock correlation matrix",
                              "sim_correlation_heatmap.png")
    plot_network(sim_graph, sim_sector_of,
                 f"Simulated market correlation network (|r| >= {args.corr_threshold}), "
                 f"community purity = {sim_purity:.2f}",
                 "sim_correlation_network.png")

    print(f"[3/4] Real network: {real_graph.number_of_nodes()} nodes, "
          f"{real_graph.number_of_edges()} edges, community-sector purity = {real_purity:.2f}")
    print(f"[3/4] Simulated network: {sim_graph.number_of_nodes()} nodes, "
          f"{sim_graph.number_of_edges()} edges, community-sector purity = {sim_purity:.2f}")

    # ---------------- 4. RMT eigenvalue comparison ----------------
    print("\n[4/4] Computing eigenvalue spectra vs Marchenko-Pastur null model...")
    real_eigs, _ = eigen_spectrum(real_returns_df.values)
    sim_eigs, _ = eigen_spectrum(sim_returns)

    real_stats = signal_vs_noise_eigenvalues(real_eigs, real_returns_df.shape[1], real_returns_df.shape[0])
    sim_stats = signal_vs_noise_eigenvalues(sim_eigs, sim_returns.shape[1], sim_returns.shape[0])

    plot_eigenvalue_comparison(
        real_eigs, sim_eigs,
        real_returns_df.shape[1], real_returns_df.shape[0],
        sim_returns.shape[1], sim_returns.shape[0],
    )

    summary_lines = [
        "SEGREGATION OF FINANCIAL MARKETS — RESULTS SUMMARY",
        "=" * 55,
        "",
        "ABM simulation",
        "-" * 15,
        f"  Agents: {args.n_agents:,} | Stocks: {model.n_stocks} | Sectors: {args.n_sectors} | Steps: {args.n_steps}",
        f"  Final segregation index: {model.segregation_series()[-1]:.3f}",
        f"  Fraction of steps in crisis regime: {model.regime_series().mean():.1%}",
        "",
        "Real market data",
        "-" * 17,
        f"  Used live S&P 500 data: {used_real}",
        f"  Assets x observations: {real_returns_df.shape[1]} x {real_returns_df.shape[0]}",
        "",
        "Correlation network community detection (purity vs true sector labels)",
        "-" * 72,
        f"  Real market:      {real_purity:.2f}",
        f"  ABM-simulated:     {sim_purity:.2f}",
        "",
        "RMT eigenvalue analysis (Marchenko-Pastur)",
        "-" * 43,
        f"  Real market   — MP upper edge: {real_stats['mp_lambda_max']:.2f} | "
        f"signal eigenvalues above bulk: {real_stats['n_signal_eigenvalues']} | "
        f"largest eigenvalue explains {real_stats['variance_explained_by_largest']:.1%} of variance",
        f"  ABM-simulated — MP upper edge: {sim_stats['mp_lambda_max']:.2f} | "
        f"signal eigenvalues above bulk: {sim_stats['n_signal_eigenvalues']} | "
        f"largest eigenvalue explains {sim_stats['variance_explained_by_largest']:.1%} of variance",
        "",
        "Files written to results/:",
        "  abm_diagnostics.png, real_correlation_heatmap.png, sim_correlation_heatmap.png,",
        "  real_correlation_network.png, sim_correlation_network.png, eigenvalue_rmt_comparison.png",
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
