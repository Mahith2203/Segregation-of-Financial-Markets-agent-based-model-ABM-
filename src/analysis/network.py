"""
Correlation network analysis
=============================
Builds a stock correlation network (nodes = stocks, edges = pairs whose
return correlation exceeds a threshold) and detects communities, which
should recover -- or approximate -- the underlying sector structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def build_correlation_graph(
    corr: pd.DataFrame, threshold: float = 0.3
) -> nx.Graph:
    """Nodes = tickers; edge (i, j) exists iff |corr(i,j)| >= threshold."""
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    tickers = corr.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            w = corr.iloc[i, j]
            if abs(w) >= threshold:
                G.add_edge(tickers[i], tickers[j], weight=float(w))
    return G


def detect_communities(G: nx.Graph) -> dict[str, int]:
    """Louvain community detection -> {ticker: community_id}."""
    try:
        import community as community_louvain  # python-louvain

        return community_louvain.best_partition(G, weight="weight", random_state=42)
    except ImportError:
        # Fallback: NetworkX's built-in greedy modularity communities
        comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
        partition = {}
        for cid, nodes in enumerate(comms):
            for node in nodes:
                partition[node] = cid
        return partition


def community_sector_agreement(
    partition: dict[str, int], sector_of: dict[str, str]
) -> float:
    """Rough measure of how well detected communities line up with true
    sectors: for each detected community, take its most common true sector
    and compute the overall fraction of nodes matching their community's
    majority sector (a simple purity score in [0, 1])."""
    df = pd.DataFrame(
        {"ticker": list(partition.keys()), "community": list(partition.values())}
    )
    df["sector"] = df["ticker"].map(sector_of)
    majority = df.groupby("community")["sector"].agg(lambda s: s.value_counts().idxmax())
    df["majority_sector"] = df["community"].map(majority)
    return float((df["sector"] == df["majority_sector"]).mean())
