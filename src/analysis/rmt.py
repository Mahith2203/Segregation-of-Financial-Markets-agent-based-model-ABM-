"""
Random Matrix Theory (RMT) eigenvalue analysis
================================================
Classic approach (Laloux et al. 1999, Plerou et al. 1999) for separating
genuine cross-asset correlation structure from pure statistical noise:

  * Compute the eigenvalue spectrum of the empirical correlation matrix.
  * Compare it to the Marchenko-Pastur (MP) distribution -- the spectrum
    you'd expect from a correlation matrix built out of purely random,
    uncorrelated return series of the same dimensions.
  * Eigenvalues inside the MP bulk [lambda_min, lambda_max] are consistent
    with noise. Eigenvalues above lambda_max carry genuine information --
    typically one very large "market mode" eigenvalue plus a handful of
    smaller "sector mode" eigenvalues.

This module computes that spectrum for both the real (or real-like) market
data and the ABM-simulated data, so the two can be compared directly.
"""

from __future__ import annotations

import numpy as np


def marchenko_pastur_bounds(n_assets: int, n_obs: int) -> tuple[float, float]:
    """Upper/lower edges of the MP bulk for a T x N return matrix with
    Q = T/N (assumes unit variance, i.e. correlation matrix, not covariance)."""
    q = n_obs / n_assets
    lambda_min = (1 - np.sqrt(1 / q)) ** 2
    lambda_max = (1 + np.sqrt(1 / q)) ** 2
    return lambda_min, lambda_max


def marchenko_pastur_pdf(x: np.ndarray, n_assets: int, n_obs: int) -> np.ndarray:
    q = n_obs / n_assets
    lambda_min, lambda_max = marchenko_pastur_bounds(n_assets, n_obs)
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lambda_min) & (x <= lambda_max)
    pdf[mask] = (
        q
        / (2 * np.pi * x[mask])
        * np.sqrt((lambda_max - x[mask]) * (x[mask] - lambda_min))
    )
    return pdf


def eigen_spectrum(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    returns: (T, N) array of returns (rows = time, cols = assets).
    Returns (eigenvalues sorted ascending, correlation matrix).
    """
    corr = np.corrcoef(returns.T)
    eigvals = np.linalg.eigvalsh(corr)  # symmetric -> real eigenvalues, ascending
    return eigvals, corr


def signal_vs_noise_eigenvalues(
    eigvals: np.ndarray, n_assets: int, n_obs: int
) -> dict:
    """Split eigenvalues into 'noise band' (inside MP bulk) vs 'signal'
    (above the MP upper edge -- genuine market/sector factors)."""
    lambda_min, lambda_max = marchenko_pastur_bounds(n_assets, n_obs)
    signal = eigvals[eigvals > lambda_max]
    noise = eigvals[(eigvals >= lambda_min) & (eigvals <= lambda_max)]
    below = eigvals[eigvals < lambda_min]
    return {
        "mp_lambda_min": lambda_min,
        "mp_lambda_max": lambda_max,
        "n_signal_eigenvalues": int(len(signal)),
        "signal_eigenvalues": np.sort(signal)[::-1],
        "n_noise_eigenvalues": int(len(noise)) + int(len(below)),
        "largest_eigenvalue": float(eigvals.max()),
        "variance_explained_by_largest": float(eigvals.max() / eigvals.sum()),
    }
