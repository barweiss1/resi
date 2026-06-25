"""ReSi adapters for selected manifold representation similarity metrics.

This module is intentionally importable both inside a ReSi checkout and from
this repository alone. When ReSi is unavailable, a small local fallback keeps
the adapter classes smoke-testable with plain ``unittest``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from metrics import AlignmentMetrics

try:  # pragma: no cover - exercised on the remote ReSi checkout.
    from repsim.measures.utils import RepresentationalSimilarityMeasure
except Exception:  # pragma: no cover - covered indirectly by local tests.

    class RepresentationalSimilarityMeasure:
        """Minimal fallback matching the ReSi constructor shape used below."""

        def __init__(self, sim_func=None, **kwargs):
            self.sim_func = sim_func
            for key, value in kwargs.items():
                setattr(self, key, value)


try:  # pragma: no cover - exercised on the remote ReSi checkout.
    from repsim.measures.utils import flatten as _resi_flatten
except Exception:  # pragma: no cover - covered indirectly by local tests.
    _resi_flatten = None


MANIFOLD_RESI_MEASURE_CLASSES = [
    "ManifoldMutualKNNTop10",
    "ManifoldCKNNATop10",
    "ManifoldRBFRWKASigma05",
    "ManifoldSoftmaxRWKATemp05",
    "ManifoldRBFCKASigma05",
]


def _as_nd(R: Any, Rp: Any, shape: str):
    if _resi_flatten is not None:
        return _resi_flatten(R, Rp, shape=shape)

    if shape not in {"nd", "ntd", "nchw"}:
        raise ValueError(f"Unsupported ReSi representation shape: {shape}")

    return _flatten_one(R), _flatten_one(Rp)


def _flatten_one(x: Any):
    if torch.is_tensor(x):
        if x.ndim < 2:
            raise ValueError(f"Expected at least 2 dimensions, got shape {tuple(x.shape)}")
        return x.reshape(x.shape[0], -1)

    array = np.asarray(x)
    if array.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {array.shape}")
    return array.reshape(array.shape[0], -1)


def _num_points(R: Any) -> int:
    return int(R.shape[0])


def _topk_at_most_10(R: Any) -> int:
    n = _num_points(R)
    if n < 3:
        raise ValueError("Top-k manifold metrics require at least 3 representation rows.")
    return min(10, n - 1)


def _finite_float(value: Any, metric_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{metric_name} returned a non-finite value: {result}")
    return result


class _ManifoldMeasure(RepresentationalSimilarityMeasure):
    metric_name = None
    metric_kwargs = {}
    use_topk_10 = False

    def __init__(self):
        super().__init__(
            sim_func=self._score,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=False,
        )

    def _score(self, R, Rp, shape):
        R, Rp = _as_nd(R, Rp, shape)
        kwargs = dict(self.metric_kwargs)
        if self.use_topk_10:
            kwargs["topk"] = _topk_at_most_10(R)
        return _finite_float(
            AlignmentMetrics.measure(self.metric_name, R, Rp, **kwargs),
            self.__class__.__name__,
        )

    def __call__(self, R, Rp, shape) -> float:
        return self._score(R, Rp, shape)


class ManifoldMutualKNNTop10(_ManifoldMeasure):
    metric_name = "mutual_knn"
    use_topk_10 = True


class ManifoldCKNNATop10(_ManifoldMeasure):
    metric_name = "cknna"
    use_topk_10 = True


class ManifoldRBFRWKASigma05(_ManifoldMeasure):
    metric_name = "rbf_rwka"
    metric_kwargs = {"rbf_sigma": 0.5}


class ManifoldSoftmaxRWKATemp05(_ManifoldMeasure):
    metric_name = "softmax_rwka"
    metric_kwargs = {"temperature": 0.5}


class ManifoldRBFCKASigma05(_ManifoldMeasure):
    metric_name = "cka_rbf"
    metric_kwargs = {"rbf_sigma": 0.5}


__all__ = MANIFOLD_RESI_MEASURE_CLASSES + ["MANIFOLD_RESI_MEASURE_CLASSES"]
