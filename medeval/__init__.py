"""MedEval — a unified protocol for plugging medical eval datasets into any model.

One canonical schema decouples *datasets*, *model backends* and *metrics*, so you
get free ``N datasets × M backends × K metrics`` composition. Backends: HF (vLLM
batch), Poe, LiteLLM. See ``configs/`` for runnable examples.
"""
from __future__ import annotations

from .schema import (
    Generation, Message, Prediction, Sample, Score, TaskType,
)
from .runner import Runner, run_config
from .providers.base import available_providers, create_provider, register_provider
from .datasets.base import available_adapters, create_dataset, register_dataset
from .metrics.base import available_metrics, create_metric, register_metric
from .submit import export, to_opencompass, to_medbench
from .distributed import merge_results, run_pool, run_ray, submit_slurm
from .kg import build_classics_kg, export_kg, get_kg, load_kg
from .assets import ensure_extracted, ensure_image_base

__version__ = "0.1.0"

__all__ = [
    "Generation", "Message", "Prediction", "Sample", "Score", "TaskType",
    "Runner", "run_config",
    "available_providers", "create_provider", "register_provider",
    "available_adapters", "create_dataset", "register_dataset",
    "available_metrics", "create_metric", "register_metric",
    "export", "to_opencompass", "to_medbench",
    "merge_results", "run_pool", "run_ray", "submit_slurm",
    "build_classics_kg", "export_kg", "get_kg", "load_kg",
    "ensure_extracted", "ensure_image_base",
    "__version__",
]
