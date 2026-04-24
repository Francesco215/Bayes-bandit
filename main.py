from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from src import (
    NormalInverseGammaPrior,
    ParameterStudyResult,
    SimulationResult,
    mean_posterior_params,
    predictive_params,
    run_parameter_study,
    save_parameter_study,
    simulate_bandit,
    uncertainty_decomposition,
)


def build_demo_parser(prog: str = "main.py") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
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
    return parser


def build_study_parser(prog: str = "main.py study") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a pizza-house parameter study for several bandit algorithms.",
    )
    parser.add_argument("--runs", type=int, default=2000, help="Independent pizza testbeds.")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per independent run.")
    parser.add_argument("--houses", type=int, default=10, help="Number of pizza houses.")
    parser.add_argument("--seed", type=int, default=0, help="JAX PRNG seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for parameter_study.png and parameter_study.csv.",
    )
    parser.add_argument(
        "--mean-scale",
        type=float,
        default=1.0,
        help="Scale of normalized true house means.",
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=0.5,
        help="Minimum per-house reward noise.",
    )
    parser.add_argument(
        "--max-sigma",
        type=float,
        default=1.5,
        help="Maximum per-house reward noise.",
    )
    return parser


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


def validate_demo_args(args: argparse.Namespace) -> None:
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


def run_demo(args: argparse.Namespace) -> None:
    validate_demo_args(args)
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


def validate_study_args(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.houses < 2:
        raise ValueError("--houses must be at least 2")
    if args.mean_scale <= 0.0:
        raise ValueError("--mean-scale must be positive")
    if args.min_sigma <= 0.0:
        raise ValueError("--min-sigma must be positive")
    if args.max_sigma <= args.min_sigma:
        raise ValueError("--max-sigma must be greater than --min-sigma")


def print_study_summary(result: ParameterStudyResult, output_dir: Path) -> None:
    print("Pizza-house bandit parameter study")
    print(f"runs={result.runs} steps={result.steps} houses={result.houses}")
    print(f"output_dir={output_dir}")
    print()
    print("algorithm              parameter  best_value  average_reward")
    for curve in result.curves:
        best_index = int(jnp.argmax(curve.average_rewards))
        print(
            f"{curve.algorithm:<22} "
            f"{curve.parameter_name:<9} "
            f"{float(curve.parameter_values[best_index]):>10.5g}  "
            f"{float(curve.average_rewards[best_index]):>14.4f}"
        )


def run_study(args: argparse.Namespace) -> None:
    validate_study_args(args)
    result = run_parameter_study(
        key=jax.random.PRNGKey(args.seed),
        runs=args.runs,
        steps=args.steps,
        houses=args.houses,
        mean_scale=args.mean_scale,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
    )
    artifacts = save_parameter_study(result, args.output_dir)
    print_study_summary(result, args.output_dir)
    print()
    print(f"csv={artifacts.csv_path}")
    print(f"plot={artifacts.plot_path}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "study":
        run_study(build_study_parser().parse_args(argv[1:]))
        return
    if argv and argv[0] == "demo":
        run_demo(build_demo_parser(prog="main.py demo").parse_args(argv[1:]))
        return
    run_demo(build_demo_parser().parse_args(argv))


if __name__ == "__main__":
    main()
