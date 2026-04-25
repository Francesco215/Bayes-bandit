from __future__ import annotations

import csv
import tempfile
import unittest

import jax
import jax.numpy as jnp

from src import default_parameter_grids, run_parameter_study, save_parameter_study


class ParameterStudyTest(unittest.TestCase):
    def test_default_grids_include_all_algorithms(self) -> None:
        grids = default_parameter_grids()
        names = [grid.algorithm for grid in grids]

        self.assertEqual(
            names,
            [
                "epsilon-greedy",
                "UCB",
                "gradient bandit",
                "optimistic greedy",
                "Bayesian P(best)",
            ],
        )

        bayesian_grid = grids[-1]
        self.assertEqual(bayesian_grid.parameter_name, "kappa")
        self.assertAlmostEqual(float(bayesian_grid.values[0]), 1.0 / 128.0)
        self.assertAlmostEqual(float(bayesian_grid.values[-1]), 4.0)

    def test_temperature_sweep_grid(self) -> None:
        grids = default_parameter_grids("temperature")
        bayesian_grid = grids[-1]

        self.assertEqual(bayesian_grid.algorithm, "Bayesian P(best)")
        self.assertEqual(bayesian_grid.parameter_name, "temperature")
        self.assertAlmostEqual(float(bayesian_grid.values[0]), 1.0 / 128.0)
        self.assertAlmostEqual(float(bayesian_grid.values[-1]), 4.0)

    def test_parameter_study_shapes_and_finite_rewards(self) -> None:
        result = run_parameter_study(
            key=jax.random.PRNGKey(0),
            runs=3,
            steps=4,
            houses=3,
        )
        grids = default_parameter_grids()

        self.assertEqual(result.runs, 3)
        self.assertEqual(result.steps, 4)
        self.assertEqual(result.houses, 3)
        self.assertEqual(len(result.curves), len(grids))
        for curve, grid in zip(result.curves, grids):
            self.assertEqual(curve.algorithm, grid.algorithm)
            self.assertEqual(curve.parameter_name, grid.parameter_name)
            self.assertEqual(curve.parameter_values.shape, grid.values.shape)
            self.assertEqual(curve.average_rewards.shape, grid.values.shape)
            self.assertTrue(jnp.all(jnp.isfinite(curve.average_rewards)))

    def test_parameter_study_supports_temperature_sweep(self) -> None:
        result = run_parameter_study(
            key=jax.random.PRNGKey(2),
            runs=2,
            steps=3,
            houses=3,
            bayesian_sweep="temperature",
            bayesian_fixed_kappa=1.0,
        )
        bayesian_curve = result.curves[-1]

        self.assertEqual(bayesian_curve.algorithm, "Bayesian P(best)")
        self.assertEqual(bayesian_curve.parameter_name, "temperature")
        self.assertTrue(jnp.all(jnp.isfinite(bayesian_curve.average_rewards)))

    def test_save_parameter_study_writes_png_and_csv(self) -> None:
        result = run_parameter_study(
            key=jax.random.PRNGKey(1),
            runs=2,
            steps=3,
            houses=3,
        )
        expected_rows = sum(len(curve.parameter_values) for curve in result.curves)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = save_parameter_study(result, temp_dir)

            self.assertTrue(artifacts.csv_path.exists())
            self.assertTrue(artifacts.plot_path.exists())
            self.assertTrue(artifacts.reward_distribution_path.exists())
            self.assertGreater(artifacts.plot_path.stat().st_size, 0)
            self.assertGreater(artifacts.reward_distribution_path.stat().st_size, 0)

            with artifacts.csv_path.open() as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), expected_rows)
            self.assertEqual(
                set(rows[0].keys()),
                {"algorithm", "parameter_name", "parameter_value", "average_reward"},
            )


if __name__ == "__main__":
    unittest.main()
