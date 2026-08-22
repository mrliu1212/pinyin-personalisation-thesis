"""Boundary extension for the fixed-surface Choice Share smoothing ablation."""

from pathlib import Path

from experiments.external_memory_next import run_choice_share_smoothing_v1 as v1


v1.ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
             256.0, 512.0, 1024.0, 2048.0, 4096.0)
v1.EXPERIMENT = "choice_share_smoothing_fixed_surface_boundary_v2"
v1.RUNNER_PATH = Path(__file__)


if __name__ == "__main__":
    v1.main()
