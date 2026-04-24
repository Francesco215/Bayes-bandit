from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from src import (
    NormalInverseGammaPrior,
    SimulationResult,
    mean_posterior_params,
    predictive_params,
    simulate_bandit,
    uncertainty_decomposition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Bayesian k-pizza-house bandit simulation in JAX.",
    )
    parser.add_argument("--houses", type=int, default=6, help="Number of pizza houses.")
    parser.add_argument("--orders", type=int, default=200, help="Number of pizzas to order.")
    parser.add_argument("--seed", type=int, default=0, help="JAX PRNG seed.")
    parser.add_argument(
        "--method",
        choices=("quadrature", "monte_carlo"),
        default="quadrature",
        help="Estimator for posterior probability of being the best house.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10_000,
        help="Monte Carlo samples when --method=monte_carlo.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=2048,
        help="Quadrature grid points when --method=quadrature.",
    )
    parser.add_argument("--prior-mu", type=float, default=7.0, help="Prior mean for house quality.")
    parser.add_argument(
        "--prior-kappa",
        type=float,
        default=0.25,
        help="Prior pseudo-count strength for the latent mean.",
    )
    parser.add_argument(
        "--prior-alpha",
        type=float,
        default=2.0,
        help="Prior inverse-gamma alpha for reward variance.",
    )
    parser.add_argument(
        "--prior-beta",
        type=float,
        default=2.0,
        help="Prior inverse-gamma beta for reward variance.",
    )
    return parser.parse_args()


def _as_float(value: jax.Array) -> float:
    return float(jnp.asarray(value))


def print_result(args: argparse.Namespace, result: SimulationResult) -> None:
    mean_params = mean_posterior_params(result.posterior)
    predictive = predictive_params(result.posterior)
    aleatoric, epistemic = uncertainty_decomposition(result.posterior)

    print("Bayesian k-pizza-house simulation")
    print(f"houses={args.houses} orders={args.orders} method={args.method} seed={args.seed}")
    print(f"recommended_house={result.recommended_house}")
    print()
    print(
        "house  orders  true_mu  true_sigma  post_mu  mean_sd  pred_sd  aleatoric  epistemic  p_best"
    )
    for house in range(args.houses):
        print(
            f"{house:>5}  "
            f"{_as_float(result.stats.counts[house]):>6.0f}  "
            f"{_as_float(result.true_means[house]):>7.3f}  "
            f"{_as_float(result.true_sigmas[house]):>10.3f}  "
            f"{_as_float(mean_params.loc[house]):>7.3f}  "
            f"{_as_float(mean_params.scale[house]):>7.3f}  "
            f"{_as_float(predictive.scale[house]):>7.3f}  "
            f"{_as_float(aleatoric[house]):>9.3f}  "
            f"{_as_float(epistemic[house]):>9.3f}  "
            f"{_as_float(result.best_probabilities[house]):>6.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.houses < 1:
        raise ValueError("--houses must be at least 1")
    if args.orders < 0:
        raise ValueError("--orders must be non-negative")
    if args.mc_samples < 1:
        raise ValueError("--mc-samples must be at least 1")
    if args.grid_points < 2:
        raise ValueError("--grid-points must be at least 2")
    if args.prior_kappa <= 0.0:
        raise ValueError("--prior-kappa must be positive")
    if args.prior_alpha <= 0.0:
        raise ValueError("--prior-alpha must be positive")
    if args.prior_beta <= 0.0:
        raise ValueError("--prior-beta must be positive")

    prior = NormalInverseGammaPrior(
        mu=args.prior_mu,
        kappa=args.prior_kappa,
        alpha=args.prior_alpha,
        beta=args.prior_beta,
    )
    result = simulate_bandit(
        key=jax.random.PRNGKey(args.seed),
        num_houses=args.houses,
        orders=args.orders,
        prior=prior,
        method=args.method,
        mc_samples=args.mc_samples,
        grid_points=args.grid_points,
    )
    print_result(args, result)


if __name__ == "__main__":
    main()
