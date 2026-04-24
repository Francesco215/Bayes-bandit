from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from src import (
    NormalInverseGammaPrior,
    initial_stats,
    mean_posterior_params,
    posterior_from_stats,
    probability_best_quadrature,
    simulate_bandit,
    stats_from_observations,
    student_t_cdf,
)


class PizzaBanditTest(unittest.TestCase):
    def test_empty_stats_recover_prior_center(self) -> None:
        prior = NormalInverseGammaPrior(mu=6.5, kappa=0.5, alpha=3.0, beta=4.0)
        posterior = posterior_from_stats(initial_stats(3), prior)

        self.assertTrue(jnp.allclose(posterior.mu, jnp.array([6.5, 6.5, 6.5])))
        self.assertTrue(jnp.allclose(posterior.kappa, jnp.array([0.5, 0.5, 0.5])))
        self.assertTrue(jnp.allclose(posterior.alpha, jnp.array([3.0, 3.0, 3.0])))
        self.assertTrue(jnp.allclose(posterior.beta, jnp.array([4.0, 4.0, 4.0])))

    def test_posterior_mean_moves_toward_observed_mean(self) -> None:
        stats = stats_from_observations(
            house_indices=jnp.array([0, 0, 0, 1, 1]),
            rewards=jnp.array([9.0, 10.0, 11.0, 4.0, 5.0]),
            num_houses=2,
        )
        posterior = posterior_from_stats(stats, NormalInverseGammaPrior(mu=7.0, kappa=0.25))

        self.assertGreater(float(posterior.mu[0]), 9.5)
        self.assertLess(float(posterior.mu[1]), 5.0)

    def test_student_t_cdf_is_monotonic_and_centered(self) -> None:
        xs = jnp.array([-2.0, 0.0, 2.0])
        cdf = student_t_cdf(xs, df=5.0, loc=0.0, scale=1.0)

        self.assertLess(float(cdf[0]), float(cdf[1]))
        self.assertLess(float(cdf[1]), float(cdf[2]))
        self.assertAlmostEqual(float(cdf[1]), 0.5, places=6)

    def test_identical_houses_have_uniform_best_probability(self) -> None:
        posterior = posterior_from_stats(initial_stats(3))
        probabilities = probability_best_quadrature(posterior, grid_points=1024)

        self.assertTrue(jnp.allclose(probabilities, jnp.ones(3) / 3.0, atol=3e-2))
        self.assertAlmostEqual(float(jnp.sum(probabilities)), 1.0, places=5)

    def test_clearly_better_house_has_highest_best_probability(self) -> None:
        stats = stats_from_observations(
            house_indices=jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            rewards=jnp.array([10.0, 10.2, 9.9, 6.0, 6.1, 5.9, 7.0, 6.8, 7.2]),
            num_houses=3,
        )
        posterior = posterior_from_stats(stats)
        probabilities = probability_best_quadrature(posterior, grid_points=2048)

        self.assertEqual(int(jnp.argmax(probabilities)), 0)
        self.assertGreater(float(probabilities[0]), 0.90)

    def test_simulation_smoke(self) -> None:
        result = simulate_bandit(
            key=jax.random.PRNGKey(0),
            num_houses=4,
            orders=20,
            grid_points=512,
        )

        self.assertEqual(float(jnp.sum(result.stats.counts)), 20.0)
        self.assertEqual(result.best_probabilities.shape, (4,))
        self.assertTrue(jnp.all(jnp.isfinite(result.best_probabilities)))
        self.assertAlmostEqual(float(jnp.sum(result.best_probabilities)), 1.0, places=5)
        self.assertGreaterEqual(result.recommended_house, 0)
        self.assertLess(result.recommended_house, 4)

    def test_mean_posterior_params_are_positive_scale(self) -> None:
        params = mean_posterior_params(posterior_from_stats(initial_stats(2)))

        self.assertTrue(jnp.all(params.df > 0.0))
        self.assertTrue(jnp.all(params.scale > 0.0))


if __name__ == "__main__":
    unittest.main()
