from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class Segment:
    id: int
    label: str
    group: str
    mask: np.ndarray
    score: float = 1.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(slots=True)
class DepthPrediction:
    depth: np.ndarray
    source: str
    metric: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Layer:
    id: int
    name: str
    label: str
    group: str
    rank: int
    depth_median: float
    depth_p10: float
    depth_p90: float
    area: int
    bbox: tuple[int, int, int, int]
    alpha: np.ndarray
    rgba: np.ndarray
    albedo_rgba: np.ndarray
    shading_rgba: np.ndarray
    visible_mask: np.ndarray
    amodal_mask: np.ndarray | None = None
    source_segment_ids: list[int] = field(default_factory=list)
    occludes: list[int] = field(default_factory=list)
    occluded_by: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineOutputs:
    output_dir: Path
    manifest_path: Path
    metrics_path: Path
    ordered_layer_paths: list[Path]
    grouped_layer_paths: list[Path]
    debug_paths: dict[str, Path]
