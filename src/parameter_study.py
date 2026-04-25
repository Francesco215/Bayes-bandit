"""Parameter-study experiments for pizza-house bandit algorithms."""

from __future__ import annotations

import csv
import os
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp


BAYESIAN_TEMPERATURE = 1.0 / 64.0
BayesianSweep = Literal["kappa", "temperature"]
_VALID_BAYESIAN_SWEEPS = {"kappa", "temperature"}


class ParameterGrid(NamedTuple):
    """Parameter values to sweep for one algorithm."""

    algorithm: str
    parameter_name: str
    values: jax.Array


class StudyCurve(NamedTuple):
    """A completed parameter-study curve."""

    algorithm: str
    parameter_name: str
    parameter_values: jax.Array
    average_rewards: jax.Array


class ParameterStudyResult(NamedTuple):
    """All curves produced by a parameter study."""

    curves: tuple[StudyCurve, ...]
    runs: int
    steps: int
    houses: int
    example_true_means: jax.Array
    example_true_sigmas: jax.Array


class StudyArtifactPaths(NamedTuple):
    """Files written by a saved parameter study."""

    csv_path: Path
    plot_path: Path
    reward_distribution_path: Path


def default_parameter_grids(bayesian_sweep: BayesianSweep = "kappa") -> tuple[ParameterGrid, ...]:
    """Return Sutton/Barto-style parameter grids plus the configured Bayesian sweep."""

    if bayesian_sweep not in _VALID_BAYESIAN_SWEEPS:
        raise ValueError("bayesian_sweep must be either 'kappa' or 'temperature'")
    return (
        ParameterGrid(
            algorithm="epsilon-greedy",
            parameter_name="epsilon",
            values=2.0 ** jnp.arange(-7, -1, dtype=jnp.float32),
        ),
        ParameterGrid(
            algorithm="UCB",
            parameter_name="c",
            values=2.0 ** jnp.arange(-4, 3, dtype=jnp.float32),
        ),
        ParameterGrid(
            algorithm="gradient bandit",
            parameter_name="alpha",
            values=2.0 ** jnp.arange(-5, 2, dtype=jnp.float32),
        ),
        ParameterGrid(
            algorithm="optimistic greedy",
            parameter_name="Q0",
            values=2.0 ** jnp.arange(-2, 3, dtype=jnp.float32),
        ),
        ParameterGrid(
            algorithm="Bayesian P(best)",
            parameter_name=bayesian_sweep,
            values=2.0 ** jnp.arange(-7, 3, dtype=jnp.float32),
        ),
    )


def _random_argmax(key: jax.Array, values: jax.Array) -> jax.Array:
    max_values = jnp.max(values, axis=-1, keepdims=True)
    is_max = values == max_values
    tie_breakers = jax.random.uniform(key, shape=values.shape)
    return jnp.argmax(jnp.where(is_max, tie_breakers, -jnp.inf), axis=-1)


def _gather_by_action(values: jax.Array, actions: jax.Array) -> jax.Array:
    return jnp.take_along_axis(values, actions[:, None], axis=1).squeeze(axis=1)


def _sample_rewards(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    means = _gather_by_action(true_means, actions)
    sigmas = _gather_by_action(true_sigmas, actions)
    return means + sigmas * jax.random.normal(key, shape=actions.shape)


def _sample_average_update(
    estimates: jax.Array,
    counts: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    one_hot = jax.nn.one_hot(actions, estimates.shape[1], dtype=estimates.dtype)
    new_counts = counts + one_hot
    step_size = one_hot / jnp.maximum(new_counts, 1.0)
    new_estimates = estimates + step_size * (rewards[:, None] - estimates)
    return new_estimates, new_counts


def _environment(
    key: jax.Array,
    runs: int,
    houses: int,
    mean_scale: float,
    min_sigma: float,
    max_sigma: float,
) -> tuple[jax.Array, jax.Array]:
    means_key, sigmas_key = jax.random.split(key)
    true_means = mean_scale * jax.random.normal(means_key, shape=(runs, houses))
    true_sigmas = jax.random.uniform(
        sigmas_key,
        shape=(runs, houses),
        minval=min_sigma,
        maxval=max_sigma,
    )
    return true_means, true_sigmas


def _simulate_epsilon_parameter(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    epsilon: jax.Array,
    steps: int,
) -> jax.Array:
    runs, houses = true_means.shape
    dtype = true_means.dtype
    initial_estimates = jnp.zeros((runs, houses), dtype=dtype)
    initial_counts = jnp.zeros((runs, houses), dtype=dtype)

    def step(carry, _):
        estimates, counts, step_key = carry
        step_key, greedy_key, explore_key, random_action_key, reward_key = jax.random.split(step_key, 5)
        greedy_actions = _random_argmax(greedy_key, estimates)
        random_actions = jax.random.randint(random_action_key, shape=(runs,), minval=0, maxval=houses)
        explore = jax.random.uniform(explore_key, shape=(runs,)) < epsilon
        actions = jnp.where(explore, random_actions, greedy_actions)
        rewards = _sample_rewards(reward_key, true_means, true_sigmas, actions)
        new_estimates, new_counts = _sample_average_update(estimates, counts, actions, rewards)
        return (new_estimates, new_counts, step_key), rewards

    _, rewards = jax.lax.scan(step, (initial_estimates, initial_counts, key), None, length=steps)
    return jnp.mean(rewards)


def _simulate_ucb_parameter(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    c: jax.Array,
    steps: int,
) -> jax.Array:
    runs, houses = true_means.shape
    dtype = true_means.dtype
    initial_estimates = jnp.zeros((runs, houses), dtype=dtype)
    initial_counts = jnp.zeros((runs, houses), dtype=dtype)

    def step(carry, time_index):
        estimates, counts, step_key = carry
        step_key, action_key, reward_key = jax.random.split(step_key, 3)
        time = time_index.astype(dtype) + 1.0
        bonus = c * jnp.sqrt(jnp.log(time) / jnp.maximum(counts, 1.0))
        ucb_values = jnp.where(counts > 0.0, estimates + bonus, jnp.inf)
        actions = _random_argmax(action_key, ucb_values)
        rewards = _sample_rewards(reward_key, true_means, true_sigmas, actions)
        new_estimates, new_counts = _sample_average_update(estimates, counts, actions, rewards)
        return (new_estimates, new_counts, step_key), rewards

    _, rewards = jax.lax.scan(
        step,
        (initial_estimates, initial_counts, key),
        jnp.arange(steps),
    )
    return jnp.mean(rewards)


def _simulate_gradient_parameter(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    alpha: jax.Array,
    steps: int,
) -> jax.Array:
    runs, houses = true_means.shape
    dtype = true_means.dtype
    initial_preferences = jnp.zeros((runs, houses), dtype=dtype)
    initial_average_rewards = jnp.zeros((runs,), dtype=dtype)
    initial_count = jnp.asarray(0.0, dtype=dtype)

    def step(carry, _):
        preferences, average_rewards, count, step_key = carry
        step_key, action_key, reward_key = jax.random.split(step_key, 3)
        probabilities = jax.nn.softmax(preferences, axis=1)
        actions = jax.random.categorical(action_key, preferences, axis=1)
        rewards = _sample_rewards(reward_key, true_means, true_sigmas, actions)
        one_hot = jax.nn.one_hot(actions, houses, dtype=dtype)
        advantage = rewards - average_rewards
        new_preferences = preferences + alpha * advantage[:, None] * (one_hot - probabilities)
        new_count = count + 1.0
        new_average_rewards = average_rewards + (rewards - average_rewards) / new_count
        return (new_preferences, new_average_rewards, new_count, step_key), rewards

    _, rewards = jax.lax.scan(
        step,
        (initial_preferences, initial_average_rewards, initial_count, key),
        None,
        length=steps,
    )
    return jnp.mean(rewards)


def _simulate_optimistic_parameter(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    q0: jax.Array,
    steps: int,
    alpha: float = 0.1,
) -> jax.Array:
    runs, houses = true_means.shape
    dtype = true_means.dtype
    initial_estimates = jnp.full((runs, houses), q0, dtype=dtype)
    alpha_value = jnp.asarray(alpha, dtype=dtype)

    def step(carry, _):
        estimates, step_key = carry
        step_key, action_key, reward_key = jax.random.split(step_key, 3)
        actions = _random_argmax(action_key, estimates)
        rewards = _sample_rewards(reward_key, true_means, true_sigmas, actions)
        one_hot = jax.nn.one_hot(actions, houses, dtype=dtype)
        new_estimates = estimates + alpha_value * one_hot * (rewards[:, None] - estimates)
        return (new_estimates, step_key), rewards

    _, rewards = jax.lax.scan(step, (initial_estimates, key), None, length=steps)
    return jnp.mean(rewards)


def _simulate_bayesian_parameter(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    parameter: jax.Array,
    steps: int,
    sweep: BayesianSweep,
    fixed_kappa: jax.Array,
    fixed_temperature: jax.Array,
    prior_alpha: float = 2.0,
    prior_beta: float = 2.0,
) -> jax.Array:
    runs, houses = true_means.shape
    dtype = true_means.dtype
    prior_mu = jnp.asarray(0.0, dtype=dtype)
    parameter = jnp.asarray(parameter, dtype=dtype)
    fixed_kappa = jnp.asarray(fixed_kappa, dtype=dtype)
    fixed_temperature = jnp.asarray(fixed_temperature, dtype=dtype)
    if sweep == "kappa":
        kappa = parameter
        temperature = fixed_temperature
    else:
        kappa = fixed_kappa
        temperature = parameter
    temperature = jnp.maximum(temperature, jnp.finfo(dtype).eps)
    prior_alpha_value = jnp.asarray(prior_alpha, dtype=dtype)
    prior_beta_value = jnp.asarray(prior_beta, dtype=dtype)
    initial_counts = jnp.zeros((runs, houses), dtype=dtype)
    initial_sums = jnp.zeros((runs, houses), dtype=dtype)
    initial_sum_squares = jnp.zeros((runs, houses), dtype=dtype)

    def step(carry, _):
        counts, sums, sum_squares, step_key = carry
        step_key, sample_key, action_key, reward_key = jax.random.split(step_key, 4)
        posterior_kappa = kappa + counts
        posterior_mu = (kappa * prior_mu + sums) / posterior_kappa
        posterior_alpha = prior_alpha_value + 0.5 * counts
        posterior_beta = prior_beta_value + 0.5 * (
            sum_squares + kappa * jnp.square(prior_mu) - posterior_kappa * jnp.square(posterior_mu)
        )
        posterior_beta = jnp.maximum(posterior_beta, jnp.finfo(dtype).eps)
        df = 2.0 * posterior_alpha
        scale = jnp.sqrt(posterior_beta / (posterior_alpha * posterior_kappa))
        sampled_means = posterior_mu + scale * jax.random.t(sample_key, df, shape=(runs, houses))
        actions = jax.random.categorical(action_key, sampled_means / temperature, axis=1)
        rewards = _sample_rewards(reward_key, true_means, true_sigmas, actions)
        one_hot = jax.nn.one_hot(actions, houses, dtype=dtype)
        new_counts = counts + one_hot
        new_sums = sums + one_hot * rewards[:, None]
        new_sum_squares = sum_squares + one_hot * jnp.square(rewards[:, None])
        return (new_counts, new_sums, new_sum_squares, step_key), rewards

    _, rewards = jax.lax.scan(
        step,
        (initial_counts, initial_sums, initial_sum_squares, key),
        None,
        length=steps,
    )
    return jnp.mean(rewards)


@partial(jax.jit, static_argnames=("steps",))
def _epsilon_curve(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    parameters: jax.Array,
    steps: int,
) -> jax.Array:
    keys = jax.random.split(key, parameters.shape[0])
    return jax.vmap(_simulate_epsilon_parameter, in_axes=(0, None, None, 0, None))(
        keys,
        true_means,
        true_sigmas,
        parameters,
        steps,
    )


@partial(jax.jit, static_argnames=("steps",))
def _ucb_curve(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    parameters: jax.Array,
    steps: int,
) -> jax.Array:
    keys = jax.random.split(key, parameters.shape[0])
    return jax.vmap(_simulate_ucb_parameter, in_axes=(0, None, None, 0, None))(
        keys,
        true_means,
        true_sigmas,
        parameters,
        steps,
    )


@partial(jax.jit, static_argnames=("steps",))
def _gradient_curve(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    parameters: jax.Array,
    steps: int,
) -> jax.Array:
    keys = jax.random.split(key, parameters.shape[0])
    return jax.vmap(_simulate_gradient_parameter, in_axes=(0, None, None, 0, None))(
        keys,
        true_means,
        true_sigmas,
        parameters,
        steps,
    )


@partial(jax.jit, static_argnames=("steps",))
def _optimistic_curve(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    parameters: jax.Array,
    steps: int,
) -> jax.Array:
    keys = jax.random.split(key, parameters.shape[0])
    return jax.vmap(_simulate_optimistic_parameter, in_axes=(0, None, None, 0, None))(
        keys,
        true_means,
        true_sigmas,
        parameters,
        steps,
    )


@partial(jax.jit, static_argnames=("steps", "sweep"))
def _bayesian_curve(
    key: jax.Array,
    true_means: jax.Array,
    true_sigmas: jax.Array,
    parameters: jax.Array,
    steps: int,
    sweep: BayesianSweep,
    fixed_kappa: float,
    fixed_temperature: float,
) -> jax.Array:
    keys = jax.random.split(key, parameters.shape[0])
    return jax.vmap(
        lambda parameter_key, parameter: _simulate_bayesian_parameter(
            parameter_key,
            true_means,
            true_sigmas,
            parameter,
            steps,
            sweep,
            fixed_kappa,
            fixed_temperature,
        )
    )(keys, parameters)


def run_parameter_study(
    key: jax.Array,
    runs: int = 2000,
    steps: int = 1000,
    houses: int = 10,
    mean_scale: float = 1.0,
    min_sigma: float = 0.5,
    max_sigma: float = 1.5,
    bayesian_sweep: BayesianSweep = "kappa",
    bayesian_fixed_kappa: float = 1.0,
    bayesian_fixed_temperature: float = BAYESIAN_TEMPERATURE,
) -> ParameterStudyResult:
    """Run a normalized pizza-house parameter study."""

    if bayesian_sweep not in _VALID_BAYESIAN_SWEEPS:
        raise ValueError("bayesian_sweep must be either 'kappa' or 'temperature'")
    if bayesian_fixed_kappa <= 0.0:
        raise ValueError("bayesian_fixed_kappa must be positive")
    if bayesian_fixed_temperature <= 0.0:
        raise ValueError("bayesian_fixed_temperature must be positive")

    environment_key, epsilon_key, ucb_key, gradient_key, optimistic_key, bayesian_key = jax.random.split(
        key,
        6,
    )
    true_means, true_sigmas = _environment(
        environment_key,
        runs=runs,
        houses=houses,
        mean_scale=mean_scale,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
    )

    grids = {grid.algorithm: grid for grid in default_parameter_grids(bayesian_sweep)}
    curves = (
        StudyCurve(
            algorithm=grids["epsilon-greedy"].algorithm,
            parameter_name=grids["epsilon-greedy"].parameter_name,
            parameter_values=grids["epsilon-greedy"].values,
            average_rewards=_epsilon_curve(
                epsilon_key,
                true_means,
                true_sigmas,
                grids["epsilon-greedy"].values,
                steps,
            ),
        ),
        StudyCurve(
            algorithm=grids["UCB"].algorithm,
            parameter_name=grids["UCB"].parameter_name,
            parameter_values=grids["UCB"].values,
            average_rewards=_ucb_curve(
                ucb_key,
                true_means,
                true_sigmas,
                grids["UCB"].values,
                steps,
            ),
        ),
        StudyCurve(
            algorithm=grids["gradient bandit"].algorithm,
            parameter_name=grids["gradient bandit"].parameter_name,
            parameter_values=grids["gradient bandit"].values,
            average_rewards=_gradient_curve(
                gradient_key,
                true_means,
                true_sigmas,
                grids["gradient bandit"].values,
                steps,
            ),
        ),
        StudyCurve(
            algorithm=grids["optimistic greedy"].algorithm,
            parameter_name=grids["optimistic greedy"].parameter_name,
            parameter_values=grids["optimistic greedy"].values,
            average_rewards=_optimistic_curve(
                optimistic_key,
                true_means,
                true_sigmas,
                grids["optimistic greedy"].values,
                steps,
            ),
        ),
        StudyCurve(
            algorithm=grids["Bayesian P(best)"].algorithm,
            parameter_name=grids["Bayesian P(best)"].parameter_name,
            parameter_values=grids["Bayesian P(best)"].values,
            average_rewards=_bayesian_curve(
                bayesian_key,
                true_means,
                true_sigmas,
                grids["Bayesian P(best)"].values,
                steps,
                bayesian_sweep,
                bayesian_fixed_kappa,
                bayesian_fixed_temperature,
            ),
        ),
    )
    return ParameterStudyResult(
        curves=curves,
        runs=runs,
        steps=steps,
        houses=houses,
        example_true_means=true_means[0],
        example_true_sigmas=true_sigmas[0],
    )


def write_parameter_study_csv(result: ParameterStudyResult, path: str | Path) -> Path:
    """Write parameter-study averages to CSV."""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["algorithm", "parameter_name", "parameter_value", "average_reward"],
        )
        writer.writeheader()
        for curve in result.curves:
            for parameter_value, average_reward in zip(curve.parameter_values, curve.average_rewards):
                writer.writerow(
                    {
                        "algorithm": curve.algorithm,
                        "parameter_name": curve.parameter_name,
                        "parameter_value": float(parameter_value),
                        "average_reward": float(average_reward),
                    }
                )
    return csv_path


def plot_parameter_study(result: ParameterStudyResult, path: str | Path) -> Path:
    """Save a Sutton/Barto-style parameter-study plot."""

    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_config = plot_path.parent / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    cache_dir = plot_path.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "epsilon-greedy": {"color": "#ff2d1f", "marker": "o"},
        "UCB": {"color": "#1f55ff", "marker": "s"},
        "gradient bandit": {"color": "#00b52a", "marker": "^"},
        "optimistic greedy": {"color": "#111111", "marker": "D"},
        "Bayesian P(best)": {"color": "#8a2be2", "marker": "P"},
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for curve in result.curves:
        style = styles.get(curve.algorithm, {})
        ax.plot(
            curve.parameter_values,
            curve.average_rewards,
            label=f"{curve.algorithm} ({curve.parameter_name})",
            linewidth=2.0,
            markersize=5.0,
            **style,
        )

    tick_values = 2.0 ** jnp.arange(-7, 3, dtype=jnp.float32)
    tick_labels = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]
    ax.set_xscale("log", base=2)
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("parameter value")
    ax.set_ylabel(f"average reward over first {result.steps} steps")
    ax.set_title("Pizza-house bandit parameter study")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _prepare_matplotlib(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_config = output_path.parent / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    return output_path


def plot_reward_distributions(
    true_means: jax.Array,
    true_sigmas: jax.Array,
    path: str | Path,
    title: str = "Reward distributions",
) -> Path:
    """Save a Sutton/Barto-style plot of the true per-house reward distributions."""

    plot_path = _prepare_matplotlib(path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = jnp.asarray(true_means)
    sigmas = jnp.asarray(true_sigmas)
    houses = int(means.shape[0])
    min_y = float(jnp.min(means - 3.5 * sigmas))
    max_y = float(jnp.max(means + 3.5 * sigmas))
    y_values = jnp.linspace(min_y, max_y, 512)
    x_positions = jnp.arange(1, houses + 1)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.72 * houses + 2.5), 5.8))
    for index in range(houses):
        density = jnp.exp(
            -0.5 * jnp.square((y_values - means[index]) / sigmas[index])
        ) / (sigmas[index] * jnp.sqrt(2.0 * jnp.pi))
        width = 0.42 * density / jnp.max(density)
        x_center = x_positions[index]
        ax.fill_betweenx(
            y_values,
            x_center - width,
            x_center + width,
            color="0.62",
            linewidth=0,
        )
        ax.plot(
            [x_center - 0.22, x_center + 0.22],
            [means[index], means[index]],
            color="0.1",
            linewidth=0.8,
        )
        ax.text(
            x_center + 0.25,
            means[index],
            rf"$q_{{*}}({index + 1})$",
            va="center",
            fontsize=9,
        )

    ax.axhline(0.0, color="0.25", linestyle=(0, (6, 6)), linewidth=1.0)
    ax.set_xlim(0.5, houses + 0.5)
    ax.set_xticks(x_positions)
    ax.set_xlabel("Pizza house")
    ax.set_ylabel("Reward distribution")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="0.88", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_parameter_study(result: ParameterStudyResult, output_dir: str | Path) -> StudyArtifactPaths:
    """Save CSV and plot artifacts for a parameter-study result."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = write_parameter_study_csv(result, output_path / "parameter_study.csv")
    plot_path = plot_parameter_study(result, output_path / "parameter_study.png")
    reward_distribution_path = plot_reward_distributions(
        result.example_true_means,
        result.example_true_sigmas,
        output_path / "reward_distributions.png",
        title="Example pizza-house reward distributions",
    )
    return StudyArtifactPaths(
        csv_path=csv_path,
        plot_path=plot_path,
        reward_distribution_path=reward_distribution_path,
    )
