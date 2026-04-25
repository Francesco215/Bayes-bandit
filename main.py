from __future__ import annotations

import sys
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

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


def _as_float(value: jax.Array) -> float:
    return float(jnp.asarray(value))


def print_result(cfg: DictConfig, result: SimulationResult) -> None:
    mean_params = mean_posterior_params(result.posterior)
    predictive = predictive_params(result.posterior)
    aleatoric, epistemic = uncertainty_decomposition(result.posterior)

    print("Bayesian k-pizza-house simulation")
    print(f"houses={cfg.houses} orders={cfg.orders} method={cfg.method} seed={cfg.seed}")
    print(f"recommended_house={result.recommended_house}")
    print()
    print(
        "house  orders  true_mu  true_sigma  post_mu  mean_sd  pred_sd  aleatoric  epistemic  p_best"
    )
    for house in range(cfg.houses):
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


def validate_demo_config(cfg: DictConfig) -> None:
    if cfg.houses < 1:
        raise ValueError("demo.houses must be at least 1")
    if cfg.orders < 0:
        raise ValueError("demo.orders must be non-negative")
    if cfg.mc_samples < 1:
        raise ValueError("demo.mc_samples must be at least 1")
    if cfg.grid_points < 2:
        raise ValueError("demo.grid_points must be at least 2")
    if cfg.prior.kappa <= 0.0:
        raise ValueError("demo.prior.kappa must be positive")
    if cfg.prior.alpha <= 0.0:
        raise ValueError("demo.prior.alpha must be positive")
    if cfg.prior.beta <= 0.0:
        raise ValueError("demo.prior.beta must be positive")


def run_demo(cfg: DictConfig) -> None:
    validate_demo_config(cfg)
    prior = NormalInverseGammaPrior(
        mu=cfg.prior.mu,
        kappa=cfg.prior.kappa,
        alpha=cfg.prior.alpha,
        beta=cfg.prior.beta,
    )
    result = simulate_bandit(
        key=jax.random.PRNGKey(cfg.seed),
        num_houses=cfg.houses,
        orders=cfg.orders,
        prior=prior,
        method=cfg.method,
        mc_samples=cfg.mc_samples,
        grid_points=cfg.grid_points,
    )
    print_result(cfg, result)


def validate_study_config(cfg: DictConfig) -> None:
    if cfg.runs < 1:
        raise ValueError("study.runs must be at least 1")
    if cfg.steps < 1:
        raise ValueError("study.steps must be at least 1")
    if cfg.houses < 2:
        raise ValueError("study.houses must be at least 2")
    if cfg.mean_scale <= 0.0:
        raise ValueError("study.mean_scale must be positive")
    if cfg.min_sigma <= 0.0:
        raise ValueError("study.min_sigma must be positive")
    if cfg.max_sigma <= cfg.min_sigma:
        raise ValueError("study.max_sigma must be greater than study.min_sigma")
    if cfg.bayesian_sweep not in {"kappa", "temperature"}:
        raise ValueError("study.bayesian_sweep must be either 'kappa' or 'temperature'")
    if cfg.bayesian_fixed_kappa <= 0.0:
        raise ValueError("study.bayesian_fixed_kappa must be positive")
    if cfg.bayesian_fixed_temperature <= 0.0:
        raise ValueError("study.bayesian_fixed_temperature must be positive")


def print_study_summary(result: ParameterStudyResult, output_dir: Path) -> None:
    print("Pizza-house bandit parameter study")
    print(f"runs={result.runs} steps={result.steps} houses={result.houses}")
    print(f"output_dir={output_dir}")
    print()
    print("algorithm              parameter    best_value  average_reward")
    for curve in result.curves:
        best_index = int(jnp.argmax(curve.average_rewards))
        print(
            f"{curve.algorithm:<22} "
            f"{curve.parameter_name:<11} "
            f"{float(curve.parameter_values[best_index]):>10.5g}  "
            f"{float(curve.average_rewards[best_index]):>14.4f}"
        )


def run_study(cfg: DictConfig) -> None:
    validate_study_config(cfg)
    output_dir = Path(str(cfg.output_dir))
    result = run_parameter_study(
        key=jax.random.PRNGKey(cfg.seed),
        runs=cfg.runs,
        steps=cfg.steps,
        houses=cfg.houses,
        mean_scale=cfg.mean_scale,
        min_sigma=cfg.min_sigma,
        max_sigma=cfg.max_sigma,
        bayesian_sweep=cfg.bayesian_sweep,
        bayesian_fixed_kappa=cfg.bayesian_fixed_kappa,
        bayesian_fixed_temperature=cfg.bayesian_fixed_temperature,
    )
    artifacts = save_parameter_study(result, output_dir)
    print_study_summary(result, output_dir)
    print()
    print(f"csv={artifacts.csv_path}")
    print(f"plot={artifacts.plot_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == "demo":
        run_demo(cfg.demo)
        return
    if cfg.mode == "study":
        run_study(cfg.study)
        return
    raise ValueError("mode must be either 'demo' or 'study'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"demo", "study"}:
        sys.argv[1] = f"mode={sys.argv[1]}"
    main()
