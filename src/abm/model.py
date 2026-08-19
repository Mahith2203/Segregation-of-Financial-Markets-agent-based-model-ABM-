"""
Agent-Based Model of Herding & Segregation in Financial Markets
=================================================================

Design
------
* N agents sit on a sparse random interaction network (their "neighbors").
* There are N_SECTORS sectors, each containing STOCKS_PER_SECTOR stocks
  (default 5 x 10 = 50 stocks).
* Each agent is assigned to exactly one stock at a time ("home stock"),
  analogous to a household's location in Schelling's segregation model.
* Each time step, agents form a trading opinion (buy / sell / hold) that is
  a mix of:
    1. a private idiosyncratic signal (noise trader behaviour), and
    2. the average opinion of their network neighbors (herding),
  weighted by a herding strength `h`.
* Net demand for each stock (buy-fraction minus sell-fraction among the
  agents currently assigned to it) drives that stock's return for the step,
  plus a small fundamental noise term.
* Agents periodically re-evaluate satisfaction with their current stock:
  if the fraction of neighbors sharing their stock/sector falls below a
  tolerance threshold, they probabilistically migrate to a different stock
  (a Schelling-style move). This produces emergent *segregation* -- agents
  cluster with like-minded neighbors on the interaction network, which in
  turn produces cross-stock/sector correlation and volatility clustering.

Everything is vectorized with NumPy/SciPy sparse operations so it scales to
100,000+ agents on a laptop.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from dataclasses import dataclass, field


@dataclass
class ABMConfig:
    n_agents: int = 100_000
    n_sectors: int = 5
    stocks_per_sector: int = 10
    avg_intra_stock_degree: int = 8    # dense ties within an agent's own stock community
    avg_intra_sector_degree: int = 3   # looser ties across stocks in the same sector
    avg_market_degree: int = 1         # sparse market-wide weak ties (all sectors)
    herding_strength: float = 0.65     # weight on neighbor opinion vs private signal
    tolerance: float = 0.35            # min fraction of like neighbors before agent is "unhappy"
    migration_prob: float = 0.08       # prob. an unhappy agent migrates in a given step
    fundamental_vol: float = 0.004     # idiosyncratic/fundamental noise on returns
    price_impact: float = 0.02         # sensitivity of returns to net order imbalance

    # -- shared factor structure (drives realistic sector/market co-movement) --
    market_vol: float = 0.003          # common market-wide shock each step
    sector_vol: float = 0.0025         # common shock shared within each sector

    # -- calm/crisis volatility regime (Markov switching) --
    p_calm_to_crisis: float = 0.02
    p_crisis_to_calm: float = 0.12
    crisis_vol_multiplier: float = 3.5   # fundamental/market/sector vol scale-up in crisis
    crisis_herding_boost: float = 0.20   # herding gets stronger under panic

    seed: int | None = 42

    @property
    def n_stocks(self) -> int:
        return self.n_sectors * self.stocks_per_sector


class MarketABM:
    """Vectorized herding/segregation agent-based market model."""

    def __init__(self, config: ABMConfig = ABMConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.n = config.n_agents
        self.n_stocks = config.n_stocks

        # stock -> sector lookup (stock ids grouped contiguously by sector)
        self.stock_sector = np.repeat(
            np.arange(config.n_sectors), config.stocks_per_sector
        )

        # Agents start uniformly distributed across stocks
        self.agent_stock = self.rng.integers(0, self.n_stocks, size=self.n)

        # Continuous sentiment state per agent (real-valued). Buy/sell/hold is
        # derived from this each step only for the order-flow/return calc; the
        # continuous value (not the discretized action) is what propagates
        # through herding. This keeps a constant trickle of fresh randomness
        # in the system each period, which is what prevents the classic
        # voter-model pathology of collapsing to one global absorbing state
        # (everyone permanently buying or permanently selling).
        self.sentiment = self.rng.normal(0, 1, size=self.n)

        # The interaction network has genuine community structure: agents are
        # densely tied to a "home" stock-community (their initial agent_stock),
        # more loosely tied to other communities in the same sector, and only
        # sparsely tied to the rest of the market. This is what allows *multiple*
        # coexisting clusters to emerge from herding instead of the whole
        # population collapsing onto one stock (a well-known pathology of
        # unstructured voter-model herding).
        self.adj = self._build_hierarchical_graph(
            self.n,
            home_community=self.agent_stock,
            community_sector=self.stock_sector,
            avg_intra=config.avg_intra_stock_degree,
            avg_sector=config.avg_intra_sector_degree,
            avg_market=config.avg_market_degree,
            rng=self.rng,
        )
        self.degree = np.asarray(self.adj.sum(axis=1)).flatten()
        self.degree[self.degree == 0] = 1  # avoid div-by-zero for isolated nodes

        self.regime = 0  # 0 = calm, 1 = crisis

        self.history: dict[str, list] = {
            "returns": [],
            "segregation_index": [],
            "avg_abs_opinion": [],
            "regime": [],
        }

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _random_pairs_within(idx: np.ndarray, n_edges: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """n_edges random (src, dst) pairs drawn from the index pool `idx`."""
        if len(idx) < 2 or n_edges <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        s = idx[rng.integers(0, len(idx), size=n_edges)]
        d = idx[rng.integers(0, len(idx), size=n_edges)]
        keep = s != d
        return s[keep], d[keep]

    @classmethod
    def _build_hierarchical_graph(
        cls,
        n_agents: int,
        home_community: np.ndarray,
        community_sector: np.ndarray,
        avg_intra: int,
        avg_sector: int,
        avg_market: int,
        rng,
    ) -> sparse.csr_matrix:
        """
        Builds a network with 3 nested tiers of connection density:
          1. intra-community  (dense)  -- agents sharing the same home stock
          2. intra-sector     (medium) -- agents in different stocks of the same sector
          3. market-wide      (sparse) -- fully random ties across the whole population

        This mirrors real trading-network structure (tight circles around a given
        stock's community, looser sector-level information flow, weak market-wide
        contagion) and is what lets herding produce *multiple* coexisting clusters
        instead of collapsing to a single global consensus.
        """
        n_communities = int(home_community.max()) + 1
        agent_sector = community_sector[home_community]
        n_sectors = int(community_sector.max()) + 1

        all_src, all_dst = [], []

        # Tier 1: dense intra-stock-community ties
        for c in range(n_communities):
            idx = np.where(home_community == c)[0]
            n_e = max(1, len(idx) * avg_intra // 2)
            s, d = cls._random_pairs_within(idx, n_e, rng)
            all_src.append(s)
            all_dst.append(d)

        # Tier 2: looser same-sector ties (spans multiple stock-communities)
        for sec in range(n_sectors):
            idx = np.where(agent_sector == sec)[0]
            n_e = max(1, len(idx) * avg_sector // 2)
            s, d = cls._random_pairs_within(idx, n_e, rng)
            all_src.append(s)
            all_dst.append(d)

        # Tier 3: sparse market-wide weak ties
        n_e = max(1, n_agents * avg_market // 2)
        s, d = cls._random_pairs_within(np.arange(n_agents), n_e, rng)
        all_src.append(s)
        all_dst.append(d)

        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        data = np.ones(len(rows), dtype=np.float32)

        adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_agents, n_agents))
        adj.data[:] = 1.0
        adj.sum_duplicates()
        return adj

    # ------------------------------------------------------------------ #
    # Simulation step
    # ------------------------------------------------------------------ #
    def _step(self) -> tuple[np.ndarray, float, float]:
        cfg = self.cfg

        # --- 0. Volatility regime (Markov switching calm <-> crisis) ---
        if self.regime == 0 and self.rng.random() < cfg.p_calm_to_crisis:
            self.regime = 1
        elif self.regime == 1 and self.rng.random() < cfg.p_crisis_to_calm:
            self.regime = 0
        vol_mult = cfg.crisis_vol_multiplier if self.regime == 1 else 1.0
        herding = cfg.herding_strength + (
            cfg.crisis_herding_boost if self.regime == 1 else 0.0
        )
        herding = min(herding, 0.95)

        # --- 1. Herding: neighbor average sentiment (continuous) ---
        neighbor_sum = self.adj @ self.sentiment
        neighbor_avg = neighbor_sum / self.degree

        # --- 2. Private signal (fresh idiosyncratic noise-trader belief) ---
        private_signal = self.rng.normal(0, 1, size=self.n)

        # --- 3. Update continuous sentiment (this is what propagates) ---
        self.sentiment = herding * neighbor_avg + (1 - herding) * private_signal

        # --- 4. Discretize into buy/sell/hold for this step's order flow only ---
        opinion = np.zeros(self.n, dtype=np.int8)
        opinion[self.sentiment > 0.15] = 1
        opinion[self.sentiment < -0.15] = -1
        self.opinion = opinion  # kept for reporting / segregation-adjacent stats

        buy = (opinion == 1).astype(np.float32)
        sell = (opinion == -1).astype(np.float32)

        stock_buy = np.bincount(self.agent_stock, weights=buy, minlength=self.n_stocks)
        stock_sell = np.bincount(self.agent_stock, weights=sell, minlength=self.n_stocks)
        stock_count = np.bincount(self.agent_stock, minlength=self.n_stocks)
        stock_count[stock_count == 0] = 1

        net_imbalance = (stock_buy - stock_sell) / stock_count

        # --- 5. Shared factor structure: market-wide + sector-wide common shocks ---
        # This is what produces realistic cross-stock correlation (a market
        # factor plus sector factors), and the crisis-regime vol_mult is what
        # produces the "correlations rise in high-volatility periods" effect.
        idiosyncratic = self.rng.normal(0, cfg.fundamental_vol * vol_mult, size=self.n_stocks)
        market_shock = self.rng.normal(0, cfg.market_vol * vol_mult)
        sector_shock = self.rng.normal(0, cfg.sector_vol * vol_mult, size=cfg.n_sectors)

        returns = (
            cfg.price_impact * net_imbalance
            + idiosyncratic
            + market_shock
            + sector_shock[self.stock_sector]
        )

        # --- 5. Schelling-style migration (segregation dynamics) ---
        # neighbor_stock_counts[i, k] = number of neighbors of agent i on stock k
        onehot = sparse.eye(self.n_stocks, format="csr")[self.agent_stock]
        neighbor_stock_counts = self.adj @ onehot  # sparse (n, n_stocks)

        own_stock_frac = np.asarray(
            neighbor_stock_counts[np.arange(self.n), self.agent_stock]
        ).flatten() / self.degree

        unhappy = own_stock_frac < cfg.tolerance
        will_move = unhappy & (self.rng.random(self.n) < cfg.migration_prob)
        n_move = int(will_move.sum())
        if n_move:
            move_idx = np.where(will_move)[0]
            # Agents move toward whichever stock is most popular among their
            # OWN neighbors (positive feedback -> emergent clustering), with
            # a small exploration probability to avoid total lock-in.
            explore = self.rng.random(n_move) < 0.15
            counts_sub = neighbor_stock_counts[move_idx].toarray()
            has_neighbors = counts_sub.sum(axis=1) > 0
            majority_stock = np.where(
                has_neighbors, counts_sub.argmax(axis=1),
                self.rng.integers(0, self.n_stocks, size=n_move),
            )
            random_stock = self.rng.integers(0, self.n_stocks, size=n_move)
            new_stock = np.where(explore, random_stock, majority_stock)
            self.agent_stock[move_idx] = new_stock

        segregation_index = float(own_stock_frac.mean())
        avg_abs_opinion = float(np.abs(self.opinion).mean())

        return returns, segregation_index, avg_abs_opinion

    def run(self, n_steps: int = 500, verbose: bool = False) -> "MarketABM":
        for t in range(n_steps):
            returns, seg_idx, avg_op = self._step()
            self.history["returns"].append(returns)
            self.history["segregation_index"].append(seg_idx)
            self.history["avg_abs_opinion"].append(avg_op)
            self.history["regime"].append(self.regime)
            if verbose and (t + 1) % max(1, n_steps // 10) == 0:
                print(
                    f"  step {t + 1:4d}/{n_steps}  "
                    f"segregation={seg_idx:.3f}  |opinion|={avg_op:.3f}  "
                    f"regime={'CRISIS' if self.regime else 'calm'}"
                )
        return self

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def returns_matrix(self) -> np.ndarray:
        """shape (T, n_stocks)"""
        return np.vstack(self.history["returns"])

    def segregation_series(self) -> np.ndarray:
        return np.array(self.history["segregation_index"])

    def regime_series(self) -> np.ndarray:
        return np.array(self.history["regime"])

    def stock_labels(self) -> list[str]:
        sector_names = [f"S{s}" for s in range(self.cfg.n_sectors)]
        return [
            f"{sector_names[self.stock_sector[i]]}_{i:02d}"
            for i in range(self.n_stocks)
        ]
