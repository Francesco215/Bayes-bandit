"""Bayesian k-pizza-house bandit utilities implemented with JAX."""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy import special


class NormalInverseGammaPrior(NamedTuple):
    """Conjugate prior for a Gaussian with unknown mean and variance."""

    mu: float = 7.0
    kappa: float = 0.25
    alpha: float = 4.0
    beta: float = 2.0


class SufficientStats(NamedTuple):
    """Per-house sufficient statistics for Gaussian rewards."""

    counts: jax.Array
    sums: jax.Array
    sum_squares: jax.Array


class NormalInverseGammaPosterior(NamedTuple):
    """Per-house Normal-Inverse-Gamma posterior parameters."""

    mu: jax.Array
    kappa: jax.Array
    alpha: jax.Array
    beta: jax.Array


class StudentTParams(NamedTuple):
    """Location-scale Student-t parameters."""

    df: jax.Array
    loc: jax.Array
    scale: jax.Array


class SimulationResult(NamedTuple):
    """Final state returned by a synthetic bandit simulation."""

    true_means: jax.Array
    true_sigmas: jax.Array
    stats: SufficientStats
    posterior: NormalInverseGammaPosterior
    best_probabilities: jax.Array
    recommended_house: int


ProbabilityMethod = Literal["quadrature", "monte_carlo"]


def initial_stats(num_houses: int, dtype=jnp.float32) -> SufficientStats:
    """Create empty sufficient statistics for ``num_houses`` pizza houses."""

    zeros = jnp.zeros((num_houses,), dtype=dtype)
    return SufficientStats(counts=zeros, sums=zeros, sum_squares=zeros)


def stats_from_observations(
    house_indices: jax.Array,
    rewards: jax.Array,
    num_houses: int,
    dtype=jnp.float32,
) -> SufficientStats:
    """Build sufficient statistics from observed ``(house, reward)`` pairs."""

    house_indices = jnp.asarray(house_indices, dtype=jnp.int32)
    rewards = jnp.asarray(rewards, dtype=dtype)
    counts = jnp.bincount(house_indices, length=num_houses).astype(dtype)
    sums = jnp.bincount(house_indices, weights=rewards, length=num_houses)
    sum_squares = jnp.bincount(
        house_indices,
        weights=jnp.square(rewards),
        length=num_houses,
    )
    return SufficientStats(counts=counts, sums=sums, sum_squares=sum_squares)


def update_stats(
    stats: SufficientStats,
    house: int | jax.Array,
    reward: float | jax.Array,
) -> SufficientStats:
    """Add one pizza score to the sufficient statistics."""

    reward = jnp.asarray(reward, dtype=stats.sums.dtype)
    return SufficientStats(
        counts=stats.counts.at[house].add(jnp.asarray(1.0, dtype=stats.counts.dtype)),
        sums=stats.sums.at[house].add(reward),
        sum_squares=stats.sum_squares.at[house].add(jnp.square(reward)),
    )


def posterior_from_stats(
    stats: SufficientStats,
    prior: NormalInverseGammaPrior = NormalInverseGammaPrior(),
) -> NormalInverseGammaPosterior:
    """Compute per-house Normal-Inverse-Gamma posterior parameters."""

    dtype = stats.sums.dtype
    counts = stats.counts.astype(dtype)
    prior_mu = jnp.asarray(prior.mu, dtype=dtype)
    prior_kappa = jnp.asarray(prior.kappa, dtype=dtype)
    prior_alpha = jnp.asarray(prior.alpha, dtype=dtype)
    prior_beta = jnp.asarray(prior.beta, dtype=dtype)

    kappa = prior_kappa + counts
    mu = (prior_kappa * prior_mu + stats.sums) / kappa
    alpha = prior_alpha + 0.5 * counts
    beta = prior_beta + 0.5 * (
        stats.sum_squares + prior_kappa * jnp.square(prior_mu) - kappa * jnp.square(mu)
    )

    eps = jnp.finfo(dtype).eps
    return NormalInverseGammaPosterior(
        mu=mu,
        kappa=kappa,
        alpha=alpha,
        beta=jnp.maximum(beta, eps),
    )


def mean_posterior_params(posterior: NormalInverseGammaPosterior) -> StudentTParams:
    """Return parameters for the marginal posterior over each latent mean."""

    df = 2.0 * posterior.alpha
    scale = jnp.sqrt(posterior.beta / (posterior.alpha * posterior.kappa))
    return StudentTParams(df=df, loc=posterior.mu, scale=scale)


def predictive_params(posterior: NormalInverseGammaPosterior) -> StudentTParams:
    """Return parameters for the posterior predictive score distribution."""

    df = 2.0 * posterior.alpha
    scale = jnp.sqrt(
        posterior.beta * (posterior.kappa + 1.0) / (posterior.alpha * posterior.kappa)
    )
    return StudentTParams(df=df, loc=posterior.mu, scale=scale)


def uncertainty_decomposition(posterior: NormalInverseGammaPosterior) -> tuple[jax.Array, jax.Array]:
    """Return aleatoric and epistemic terms of posterior predictive variance."""

    denominator = posterior.alpha - 1.0
    finite = denominator > 0.0
    aleatoric = jnp.where(finite, posterior.beta / denominator, jnp.inf)
    epistemic = jnp.where(finite, aleatoric / posterior.kappa, jnp.inf)
    return aleatoric, epistemic


def student_t_logpdf(
    x: jax.Array,
    df: jax.Array,
    loc: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Evaluate a location-scale Student-t log density."""

    dtype = jnp.result_type(x, df, loc, scale, jnp.float32)
    x = jnp.asarray(x, dtype=dtype)
    df = jnp.asarray(df, dtype=dtype)
    loc = jnp.asarray(loc, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    z = (x - loc) / scale
    return (
        special.gammaln(0.5 * (df + 1.0))
        - special.gammaln(0.5 * df)
        - 0.5 * (jnp.log(df) + jnp.log(jnp.pi))
        - jnp.log(scale)
        - 0.5 * (df + 1.0) * jnp.log1p(jnp.square(z) / df)
    )


def student_t_pdf(
    x: jax.Array,
    df: jax.Array,
    loc: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Evaluate a location-scale Student-t density."""

    return jnp.exp(student_t_logpdf(x=x, df=df, loc=loc, scale=scale))


def student_t_cdf(
    x: jax.Array,
    df: jax.Array,
    loc: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Evaluate a location-scale Student-t CDF using JAX special functions."""

    dtype = jnp.result_type(x, df, loc, scale, jnp.float32)
    x = jnp.asarray(x, dtype=dtype)
    df = jnp.asarray(df, dtype=dtype)
    loc = jnp.asarray(loc, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    z = (x - loc) / scale
    q = df / (df + jnp.square(z))
    regularized_beta = special.betainc(0.5 * df, 0.5, jnp.clip(q, 0.0, 1.0))
    lower_tail = 0.5 * regularized_beta
    upper_tail = 1.0 - lower_tail
    return jnp.where(z >= 0.0, upper_tail, lower_tail)


def sample_student_t(
    key: jax.Array,
    params: StudentTParams,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples from independent location-scale Student-t distributions."""

    shape = sample_shape + tuple(params.loc.shape)
    standard_samples = jax.random.t(key, params.df, shape=shape)
    return params.loc + params.scale * standard_samples


def probability_best_monte_carlo(
    key: jax.Array,
    posterior: NormalInverseGammaPosterior,
    samples: int = 10_000,
) -> jax.Array:
    """Estimate ``P(k* = k | D)`` by sampling latent means."""

    params = mean_posterior_params(posterior)
    num_houses = int(params.loc.shape[0])
    if num_houses == 1:
        return jnp.ones((1,), dtype=params.loc.dtype)

    sampled_means = sample_student_t(key, params, sample_shape=(samples,))
    winners = jnp.argmax(sampled_means, axis=1)
    counts = jnp.bincount(winners, length=num_houses).astype(params.loc.dtype)
    return counts / jnp.asarray(samples, dtype=params.loc.dtype)


def probability_best_quadrature(
    posterior: NormalInverseGammaPosterior,
    grid_points: int = 2048,
    grid_width: float = 12.0,
) -> jax.Array:
    """Approximate ``P(k* = k | D)`` with the README's 1D quadrature formula."""

    params = mean_posterior_params(posterior)
    num_houses = int(params.loc.shape[0])
    dtype = params.loc.dtype
    if num_houses == 1:
        return jnp.ones((1,), dtype=dtype)

    left = jnp.min(params.loc - grid_width * params.scale)
    right = jnp.max(params.loc + grid_width * params.scale)
    xs = jnp.linspace(left, right, grid_points, dtype=dtype)

    df = params.df[:, None]
    loc = params.loc[:, None]
    scale = params.scale[:, None]
    x_grid = xs[None, :]
    pdfs = student_t_pdf(x_grid, df=df, loc=loc, scale=scale)
    cdfs = student_t_cdf(x_grid, df=df, loc=loc, scale=scale)

    house_mask = jnp.eye(num_houses, dtype=bool)[:, :, None]
    other_cdfs = jnp.where(house_mask, jnp.asarray(1.0, dtype=dtype), cdfs[None, :, :])
    other_products = jnp.prod(other_cdfs, axis=1)
    integrands = pdfs * other_products

    dx = (right - left) / jnp.asarray(grid_points - 1, dtype=dtype)
    raw = dx * (
        0.5 * integrands[:, 0]
        + jnp.sum(integrands[:, 1:-1], axis=1)
        + 0.5 * integrands[:, -1]
    )
    raw = jnp.where(jnp.isfinite(raw), raw, 0.0)
    raw = jnp.clip(raw, 0.0, jnp.inf)

    total = jnp.sum(raw)
    eps = jnp.finfo(dtype).eps
    normalized = raw / jnp.where(total > eps, total, jnp.asarray(1.0, dtype=dtype))
    fallback = jnp.ones((num_houses,), dtype=dtype) / jnp.asarray(num_houses, dtype=dtype)
    return jnp.where(total > eps, normalized, fallback)


def best_probabilities(
    key: jax.Array,
    posterior: NormalInverseGammaPosterior,
    method: ProbabilityMethod = "quadrature",
    mc_samples: int = 10_000,
    grid_points: int = 2048,
) -> jax.Array:
    """Compute posterior probabilities that each house has the best latent mean."""

    if method == "quadrature":
        return probability_best_quadrature(posterior, grid_points=grid_points)
    if method == "monte_carlo":
        return probability_best_monte_carlo(key, posterior, samples=mc_samples)
    raise ValueError(f"Unknown probability method: {method}")


def sample_house_from_best_probability(key: jax.Array, probabilities: jax.Array) -> int:
    """Sample the next house from the posterior probability-of-best vector."""

    probabilities = probabilities / jnp.sum(probabilities)
    house = jax.random.categorical(key, jnp.log(jnp.clip(probabilities, 1e-30, 1.0)))
    return int(house)


def simulate_bandit(
    key: jax.Array,
    num_houses: int = 6,
    orders: int = 200,
    prior: NormalInverseGammaPrior = NormalInverseGammaPrior(),
    method: ProbabilityMethod = "quadrature",
    mc_samples: int = 10_000,
    grid_points: int = 2048,
) -> SimulationResult:
    """Run a synthetic pizza-house bandit simulation."""

    key, means_key, sigmas_key = jax.random.split(key, 3)
    true_means = 7.0 + 1.5 * jax.random.normal(means_key, shape=(num_houses,))
    true_sigmas = jax.random.uniform(
        sigmas_key,
        shape=(num_houses,),
        minval=0.5,
        maxval=1.5,
    )

    stats = initial_stats(num_houses)
    final_probabilities = jnp.ones((num_houses,), dtype=true_means.dtype) / num_houses

    for _ in range(orders):
        key, probability_key, policy_key, reward_key = jax.random.split(key, 4)
        posterior = posterior_from_stats(stats, prior)
        final_probabilities = best_probabilities(
            probability_key,
            posterior,
            method=method,
            mc_samples=mc_samples,
            grid_points=grid_points,
        )
        house = sample_house_from_best_probability(policy_key, final_probabilities)
        reward = true_means[house] + true_sigmas[house] * jax.random.normal(reward_key)
        stats = update_stats(stats, house, reward)

    posterior = posterior_from_stats(stats, prior)
    final_probabilities = best_probabilities(
        key,
        posterior,
        method=method,
        mc_samples=mc_samples,
        grid_points=grid_points,
    )
    recommended_house = int(jnp.argmax(final_probabilities))
    return SimulationResult(
        true_means=true_means,
        true_sigmas=true_sigmas,
        stats=stats,
        posterior=posterior,
        best_probabilities=final_probabilities,
        recommended_house=recommended_house,
    )
