from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .compose import composite_layers_near_to_far, rgba_from_rgb_alpha
from .inpaint import complete_hidden_layer
from .intrinsics import intrinsic_residual, intrinsic_rgba
from .matting import refine_layer_alpha
from .semantic import BACKGROUND_GROUPS
from .types import Layer, Segment
from .utils import bbox_from_mask, image_to_float, safe_name, touches_border


@dataclass(slots=True)
class GraphEdge:
    near_id: int
    far_id: int
    confidence: float
    shared_boundary_length: int
    local_depth_gap: float
    near_local_depth: float
    far_local_depth: float
    relation: str = "in_front_of"
    overlap_ratio: float | None = None
    contact_score: float | None = None
    semantic_prior: float | None = None
    model_confidence: float | None = None
    conflict_notes: str | None = None
    reason: str = "boundary_local_depth"


@dataclass(slots=True)
class Node:
    segment: Segment
    depth_median: float
    depth_p10: float
    depth_p90: float
    depth_trimmed_mean: float
    depth_boundary_median: float
    depth_variance: float
    depth_confidence: float
    border_touch: bool
    occludes: set[int] = field(default_factory=set)
    occluded_by: set[int] = field(default_factory=set)
    outgoing_edges: dict[int, GraphEdge] = field(default_factory=dict)
    incoming_edges: dict[int, GraphEdge] = field(default_factory=dict)
    removed_edges: list[dict[str, Any]] = field(default_factory=list)


def depth_stats(mask: np.ndarray, depth: np.ndarray) -> tuple[float, float, float]:
    vals = depth[mask.astype(bool)]
    if vals.size == 0:
        return 1.0, 1.0, 1.0
    return float(np.median(vals)), float(np.percentile(vals, 10)), float(np.percentile(vals, 90))


def mask_depth_stats(mask: np.ndarray, depth: np.ndarray, *, boundary_width: int = 3) -> dict[str, float]:
    m = mask.astype(bool)
    vals = depth[m]
    if vals.size == 0:
        return {
            "median": 1.0,
            "p10": 1.0,
            "p90": 1.0,
            "trimmed_mean": 1.0,
            "boundary_median": 1.0,
            "variance": 0.0,
            "confidence": 0.0,
        }
    lo, hi = np.percentile(vals, [10, 90])
    trimmed = vals[(vals >= lo) & (vals <= hi)]
    eroded = ndi.binary_erosion(m, iterations=max(1, int(boundary_width)))
    boundary = m ^ eroded
    boundary_vals = depth[boundary]
    variance = float(np.var(vals))
    spread = float(max(0.0, hi - lo))
    confidence = float(np.clip((1.0 - spread) * np.log1p(vals.size) / 8.0, 0.0, 1.0))
    return {
        "median": float(np.median(vals)),
        "p10": float(lo),
        "p90": float(hi),
        "trimmed_mean": float(np.mean(trimmed if trimmed.size else vals)),
        "boundary_median": float(np.median(boundary_vals if boundary_vals.size else vals)),
        "variance": variance,
        "confidence": confidence,
    }


def split_stuff_by_depth(segments: list[Segment], depth: np.ndarray, bins: int, min_area: int) -> list[Segment]:
    if bins <= 1:
        return segments
    out: list[Segment] = []
    sid = 0
    for seg in segments:
        if seg.group not in BACKGROUND_GROUPS or seg.area < 3 * min_area:
            seg.id = sid
            out.append(seg)
            sid += 1
            continue
        vals = depth[seg.mask]
        if vals.size == 0:
            continue
        qs = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
        made = False
        for bi, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
            piece = seg.mask & (depth >= lo) & (depth <= hi)
            labels, n = ndi.label(piece)
            for lab in range(1, n + 1):
                cc = labels == lab
                if int(cc.sum()) >= min_area:
                    out.append(Segment(sid, f"{seg.label} plane {bi}", seg.group, cc, seg.score, bbox_from_mask(cc), seg.source, {**seg.metadata, "depth_plane": bi}))
                    sid += 1
                    made = True
        if not made:
            seg.id = sid
            out.append(seg)
            sid += 1
    return out


def _local_depth(mask: np.ndarray, other_dilated: np.ndarray, depth: np.ndarray, fallback: float) -> float:
    local = mask & other_dilated
    vals = depth[local]
    if vals.size < 8:
        vals = depth[mask]
    if vals.size == 0:
        return fallback
    return float(np.median(vals))


def _edge_confidence(gap: float, boundary_len: int, alpha_confidence: float = 1.0) -> float:
    return float(max(0.0, gap) * np.log1p(max(1, boundary_len)) * max(0.0, min(1.0, alpha_confidence)))


def edge_evidence(edge: GraphEdge) -> dict[str, float | None]:
    return {
        "boundary_depth_delta": edge.local_depth_gap,
        "overlap_ratio": edge.overlap_ratio,
        "contact_score": edge.contact_score,
        "semantic_prior": edge.semantic_prior,
        "model_confidence": edge.model_confidence if edge.model_confidence is not None else edge.confidence,
    }


def build_nodes(segments: list[Segment], depth: np.ndarray, cfg: dict[str, Any]) -> dict[int, Node]:
    nodes: dict[int, Node] = {}
    stat_boundary_width = max(1, int(cfg.get("depth_stats_boundary_width", 3)))
    for seg in segments:
        stats = mask_depth_stats(seg.mask, depth, boundary_width=stat_boundary_width)
        nodes[seg.id] = Node(
            seg,
            stats["median"],
            stats["p10"],
            stats["p90"],
            stats["trimmed_mean"],
            stats["boundary_median"],
            stats["variance"],
            stats["confidence"],
            touches_border(seg.mask, 3),
        )

    width = int(cfg.get("occlusion_boundary_width", 5))
    if width <= 0 or len(nodes) <= 1:
        return nodes

    structure = ndi.iterate_structure(ndi.generate_binary_structure(2, 2), width)
    dil = {sid: ndi.binary_dilation(n.segment.mask, structure=structure) for sid, n in nodes.items()}
    ids = list(nodes)
    threshold = float(cfg.get("occlusion_depth_threshold", 0.025))
    min_boundary = int(cfg.get("min_shared_boundary_px", 12))

    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            a, b = nodes[a_id], nodes[b_id]
            a_local_mask = a.segment.mask & dil[b_id]
            b_local_mask = b.segment.mask & dil[a_id]
            shared_boundary_len = int(max(a_local_mask.sum(), b_local_mask.sum()))
            if shared_boundary_len < min_boundary:
                continue
            za = _local_depth(a.segment.mask, dil[b_id], depth, a.depth_median)
            zb = _local_depth(b.segment.mask, dil[a_id], depth, b.depth_median)
            gap = abs(za - zb)
            overlap_ratio = float((a.segment.mask & b.segment.mask).sum() / max(1, min(a.segment.area, b.segment.area)))
            contact_score = float(np.clip(shared_boundary_len / max(1, min(a.segment.area, b.segment.area)), 0.0, 1.0))
            semantic_prior = 1.0 if a.segment.group != b.segment.group else 0.75
            if gap < threshold:
                same_plane_limit = float(cfg.get("same_plane_depth_threshold", threshold * 0.75))
                if gap <= same_plane_limit:
                    first, second = (a, b) if a.segment.id < b.segment.id else (b, a)
                    edge = GraphEdge(
                        near_id=first.segment.id,
                        far_id=second.segment.id,
                        confidence=_edge_confidence(max(same_plane_limit - gap, 0.0), shared_boundary_len),
                        shared_boundary_length=shared_boundary_len,
                        local_depth_gap=float(gap),
                        near_local_depth=float(min(za, zb)),
                        far_local_depth=float(max(za, zb)),
                        relation="same_plane",
                        overlap_ratio=overlap_ratio,
                        contact_score=contact_score,
                        semantic_prior=semantic_prior,
                        model_confidence=min(a.depth_confidence, b.depth_confidence),
                        reason="same_plane_depth",
                    )
                    first.outgoing_edges[second.segment.id] = edge
                    second.incoming_edges[first.segment.id] = edge
                continue
            near, far = (a, b) if za < zb else (b, a)
            near_z, far_z = (za, zb) if za < zb else (zb, za)
            edge = GraphEdge(
                near_id=near.segment.id,
                far_id=far.segment.id,
                confidence=_edge_confidence(gap, shared_boundary_len),
                shared_boundary_length=shared_boundary_len,
                local_depth_gap=float(gap),
                near_local_depth=float(near_z),
                far_local_depth=float(far_z),
                relation="in_front_of",
                overlap_ratio=overlap_ratio,
                contact_score=contact_score,
                semantic_prior=semantic_prior,
                model_confidence=min(near.depth_confidence, far.depth_confidence),
            )
            near.occludes.add(far.segment.id)
            far.occluded_by.add(near.segment.id)
            near.outgoing_edges[far.segment.id] = edge
            far.incoming_edges[near.segment.id] = edge
    return nodes


def _remove_weakest_cycle_edge(nodes: dict[int, Node], graph: dict[int, set[int]]) -> bool:
    weakest: tuple[float, int, int] | None = None
    for src, dsts in graph.items():
        for dst in dsts:
            edge = nodes[src].outgoing_edges.get(dst)
            conf = edge.confidence if edge is not None else 0.0
            if weakest is None or conf < weakest[0]:
                weakest = (float(conf), src, dst)
    if weakest is None:
        return False
    _, src, dst = weakest
    removed = nodes[src].outgoing_edges.get(dst)
    graph[src].discard(dst)
    nodes[src].occludes.discard(dst)
    nodes[src].outgoing_edges.pop(dst, None)
    nodes[dst].occluded_by.discard(src)
    nodes[dst].incoming_edges.pop(src, None)
    if removed is not None:
        note = asdict(removed)
        note["removed_reason"] = "cycle_resolution_weakest_edge"
        nodes[src].removed_edges.append(note)
    return True


def topo_order(nodes: dict[int, Node]) -> list[int]:
    if not nodes:
        return []
    graph = {
        sid: {
            dst for dst in set(node.occludes) & set(nodes)
            if node.outgoing_edges.get(dst) is None or node.outgoing_edges[dst].relation == "in_front_of"
        }
        for sid, node in nodes.items()
    }
    while True:
        indeg = {sid: 0 for sid in nodes}
        for src, dsts in graph.items():
            for dst in dsts:
                if dst != src:
                    indeg[dst] += 1
        q = deque(sorted([sid for sid, d in indeg.items() if d == 0], key=lambda x: nodes[x].depth_median))
        order: list[int] = []
        indeg_work = dict(indeg)
        while q:
            sid = q.popleft()
            order.append(sid)
            for far in sorted(graph[sid], key=lambda x: nodes[x].depth_median):
                indeg_work[far] -= 1
                if indeg_work[far] == 0:
                    q.append(far)
            q = deque(sorted(q, key=lambda x: nodes[x].depth_median))
        if len(order) == len(nodes):
            return order
        if not _remove_weakest_cycle_edge(nodes, graph):
            break
    return [sid for sid in sorted(nodes, key=lambda x: nodes[x].depth_median)]


def amodal_complete(mask: np.ndarray, expand_px: int) -> np.ndarray:
    cv2 = __import__("cv2")
    m = mask.astype(bool)
    if not m.any():
        return m
    ksize = max(3, expand_px | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    fill = ndi.binary_fill_holes(closed)
    hull = fill
    try:
        from skimage.morphology import convex_hull_image
        hull = convex_hull_image(fill)
    except Exception:
        pass
    expanded = ndi.binary_dilation(fill, iterations=max(1, expand_px // 3))
    return (fill | expanded | hull).astype(bool)


def edge_continuity_score(visible_mask: np.ndarray, hidden_mask: np.ndarray) -> float:
    visible = visible_mask.astype(bool)
    hidden = hidden_mask.astype(bool)
    if not hidden.any():
        return 1.0
    visible_boundary = ndi.binary_dilation(visible, iterations=1) & hidden
    hidden_boundary = ndi.binary_dilation(hidden, iterations=1) & visible
    contact = int(visible_boundary.sum() + hidden_boundary.sum())
    hidden_edge = int((hidden ^ ndi.binary_erosion(hidden)).sum())
    return float(np.clip(contact / max(1, hidden_edge), 0.0, 1.0))


def resolve_amodal_mask(mask: np.ndarray, cfg: dict[str, Any], expand_px: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    method = str(cfg.get("method", "heuristic")).lower()
    if method in {"none", "off", "disabled"}:
        return None, {"method": method, "backend_used": False}
    if method in {"auto", "heuristic"}:
        return amodal_complete(mask, expand_px), {"method": method, "backend": "heuristic", "backend_used": False, "fallback_used": method == "auto"}
    if method in {"sameo", "external"}:
        # Optional model/external amodal completion is registered by doctor; the
        # native runtime keeps a deterministic CPU fallback unless a backend is
        # wired through config in a later adapter.
        return amodal_complete(mask, expand_px), {
            "method": method,
            "backend": "heuristic",
            "backend_used": False,
            "fallback_used": True,
            "warning": f"{method} amodal backend unavailable in native runtime",
        }
    return amodal_complete(mask, expand_px), {"method": method, "backend": "heuristic", "backend_used": False, "fallback_used": True}


def visible_masks_by_order(segments: list[Segment], order: list[int]) -> dict[int, np.ndarray]:
    by_id = {s.id: s for s in segments}
    occupied = np.zeros_like(segments[0].mask, dtype=bool) if segments else np.zeros((1, 1), dtype=bool)
    visible: dict[int, np.ndarray] = {}
    for sid in order:
        m = by_id[sid].mask & ~occupied
        visible[sid] = m
        occupied |= m
    return visible


def _base_label(label: str) -> str:
    low = str(label).lower()
    for token in ("background visible", "background completed", " plane "):
        if token in low:
            low = low.split(token)[0]
    return safe_name(low)


def renumber_layers_in_place(layers: list[Layer]) -> list[Layer]:
    ordered = sorted(layers, key=lambda x: (x.rank, x.depth_median, -x.area))
    for idx, layer in enumerate(ordered):
        layer.id = idx
        layer.rank = idx
        group_name = safe_name(layer.group)
        base_name = _base_label(layer.label)
        layer.name = f"{idx:03d}_{group_name}" if base_name == group_name else f"{idx:03d}_{group_name}_{base_name}"
    return ordered


def _layer_color_signature(layer: Layer) -> np.ndarray:
    mask = layer.alpha > 0.05
    if not mask.any():
        return np.zeros(3, dtype=np.float32)
    vals = image_to_float(layer.rgba[..., :3])[mask]
    return np.median(vals, axis=0).astype(np.float32)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0, max(ax0 - bx1, bx0 - ax1))
    dy = max(0, max(ay0 - by1, by0 - ay1))
    return int(max(dx, dy))


def _masks_near(a: np.ndarray, b: np.ndarray, gap_px: int) -> bool:
    if gap_px <= 0:
        return bool((a & b).any())
    dil = ndi.binary_dilation(a, iterations=gap_px)
    return bool((dil & b).any())


def _merge_bucket_candidate(bucket: list[Layer], layer: Layer, cfg: dict[str, Any]) -> bool:
    first = bucket[0]
    if first.group == "background" or layer.group == "background":
        return False
    if first.group != layer.group:
        return False
    depth_thresh = float(cfg.get("merge_depth_threshold", 0.04))
    color_thresh = float(cfg.get("merge_color_threshold", 0.17))
    spatial_gap = int(cfg.get("merge_spatial_gap_px", 20))
    bucket_depth = float(np.median([b.depth_median for b in bucket]))
    if abs(bucket_depth - layer.depth_median) > depth_thresh:
        return False
    bucket_color = np.median(np.stack([_layer_color_signature(b) for b in bucket], axis=0), axis=0)
    if float(np.linalg.norm(bucket_color - _layer_color_signature(layer))) > color_thresh:
        return False
    if layer.group in BACKGROUND_GROUPS:
        return True
    label_match = _base_label(first.label) == _base_label(layer.label)
    if not label_match:
        return False
    merged_mask = np.maximum.reduce([b.alpha for b in bucket]) > 0.05
    return _masks_near(merged_mask, layer.alpha > 0.05, spatial_gap) or _bbox_gap(first.bbox, layer.bbox) <= spatial_gap


def _merge_layer_bucket(bucket: list[Layer], rank: int) -> Layer:
    ordered = sorted(bucket, key=lambda l: l.rank)
    if len(ordered) == 1:
        layer = ordered[0]
        layer.rank = rank
        layer.id = rank
        return layer
    rgba = composite_layers_near_to_far(ordered)
    albedo = composite_layers_near_to_far(
        [Layer(l.id, l.name, l.label, l.group, l.rank, l.depth_median, l.depth_p10, l.depth_p90, l.area, l.bbox, l.alpha, l.albedo_rgba, l.albedo_rgba, l.shading_rgba, l.visible_mask, l.amodal_mask, l.source_segment_ids, l.occludes, l.occluded_by, l.metadata) for l in ordered]
    )
    shading = composite_layers_near_to_far(
        [Layer(l.id, l.name, l.label, l.group, l.rank, l.depth_median, l.depth_p10, l.depth_p90, l.area, l.bbox, l.alpha, l.shading_rgba, l.albedo_rgba, l.shading_rgba, l.visible_mask, l.amodal_mask, l.source_segment_ids, l.occludes, l.occluded_by, l.metadata) for l in ordered]
    )
    alpha = image_to_float(rgba)[..., 3]
    visible = alpha > 0.05
    group = ordered[0].group
    labels = {_base_label(l.label) for l in ordered}
    label = ordered[0].label if len(labels) == 1 else f"{group} merged"
    merged_amodal = None
    if any(l.amodal_mask is not None for l in ordered):
        merged_amodal = np.zeros_like(ordered[0].visible_mask, dtype=bool)
        for layer in ordered:
            if layer.amodal_mask is not None:
                merged_amodal |= layer.amodal_mask.astype(bool)
    merged_hidden = None
    if any(l.hidden_mask is not None for l in ordered):
        merged_hidden = np.zeros_like(ordered[0].visible_mask, dtype=bool)
        for layer in ordered:
            if layer.hidden_mask is not None:
                merged_hidden |= layer.hidden_mask.astype(bool)
    completed = composite_layers_near_to_far(
        [
            Layer(l.id, l.name, l.label, l.group, l.rank, l.depth_median, l.depth_p10, l.depth_p90, l.area, l.bbox, l.alpha, l.completed_rgba if l.completed_rgba is not None else l.rgba, l.albedo_rgba, l.shading_rgba, l.visible_mask, l.amodal_mask, l.source_segment_ids, l.occludes, l.occluded_by, l.metadata)
            for l in ordered
        ]
    )
    source_ids = sorted({sid for l in ordered for sid in l.source_segment_ids})
    occludes = sorted({sid for l in ordered for sid in l.occludes})
    occluded_by = sorted({sid for l in ordered for sid in l.occluded_by})
    depth_weights = np.array([max(1, l.area) for l in ordered], dtype=np.float32)
    depth_values = np.array([l.depth_median for l in ordered], dtype=np.float32)
    p10_values = np.array([l.depth_p10 for l in ordered], dtype=np.float32)
    p90_values = np.array([l.depth_p90 for l in ordered], dtype=np.float32)
    meta = {
        "merged_members": [l.name for l in ordered],
        "merged_count": len(ordered),
    }
    return Layer(
        rank,
        f"{rank:03d}_{safe_name(group)}_{safe_name(label)}",
        label,
        group,
        rank,
        float(np.average(depth_values, weights=depth_weights)),
        float(np.min(p10_values)),
        float(np.max(p90_values)),
        int(visible.sum()),
        bbox_from_mask(visible),
        alpha,
        rgba,
        albedo,
        shading,
        visible,
        merged_amodal,
        source_ids,
        occludes,
        occluded_by,
        {**ordered[0].metadata, **meta},
        hidden_mask=merged_hidden,
        completed_rgba=completed,
    )


def merge_compatible_layers(layers: list[Layer], cfg: dict[str, Any]) -> list[Layer]:
    if not layers or not bool(cfg.get("merge_enabled", True)):
        return renumber_layers_in_place(list(layers))
    buckets: list[list[Layer]] = []
    for layer in sorted(layers, key=lambda l: l.rank):
        if layer.group == "background" or not buckets:
            buckets.append([layer])
            continue
        if _merge_bucket_candidate(buckets[-1], layer, cfg):
            buckets[-1].append(layer)
        else:
            buckets.append([layer])
    merged = [_merge_layer_bucket(bucket, rank) for rank, bucket in enumerate(buckets)]
    return renumber_layers_in_place(merged)


def build_layers(
    rgb: np.ndarray,
    segments: list[Segment],
    depth: np.ndarray,
    albedo: np.ndarray,
    shading: np.ndarray,
    cfg: dict[str, Any],
    matting_cfg: dict[str, Any],
    *,
    device: str = "auto",
    amodal_cfg: dict[str, Any] | None = None,
    inpainting_cfg: dict[str, Any] | None = None,
) -> tuple[list[Layer], dict[int, Node]]:
    h, w = depth.shape
    min_area = max(12, int(h * w * float(cfg.get("min_layer_area_ratio", 0.0015))))
    segments = [s for s in segments if s.area >= min_area]
    segments = split_stuff_by_depth(segments, depth, int(cfg.get("split_stuff_depth_bins", 3)), min_area)
    if len(segments) > int(cfg.get("max_layers", 64)):
        segments = sorted(segments, key=lambda s: s.area, reverse=True)[: int(cfg.get("max_layers", 64))]
        for i, s in enumerate(segments):
            s.id = i
    nodes = build_nodes(segments, depth, cfg)
    ordering_method = str(cfg.get("ordering_method", "boundary")).lower()
    ordering_scores: dict[int, float] = {}
    if ordering_method in {"learned", "ranker"}:
        from .ranker import learned_order, load_ranker

        model_path = str(cfg.get("ranker_model_path", "")).strip()
        if not model_path:
            raise RuntimeError("Learned ordering requires layering.ranker_model_path")
        order, ordering_scores = learned_order(nodes, load_ranker(model_path))
    else:
        order = topo_order(nodes)
    visible = visible_masks_by_order(segments, order)
    layers: list[Layer] = []
    for rank, sid in enumerate(order):
        seg = next(s for s in segments if s.id == sid)
        vis = visible.get(sid, seg.mask)
        if int(vis.sum()) < min_area:
            continue
        node = nodes[sid]
        alpha, alpha_meta = refine_layer_alpha(rgb, vis, depth, matting_cfg, device=device)
        rgba = rgba_from_rgb_alpha(rgb, alpha)
        ar, sr = intrinsic_rgba(albedo, shading, alpha)
        iid_residual = intrinsic_residual(rgb, albedo, shading, vis)
        amodal_meta: dict[str, Any] = {"method": "none", "backend_used": False}
        if bool(cfg.get("amodal_enabled", True)):
            amodal, amodal_meta = resolve_amodal_mask(vis, amodal_cfg or {"method": "heuristic"}, int(cfg.get("amodal_expand_px", 16)))
        else:
            amodal = None
        hidden = (amodal.astype(bool) & ~vis.astype(bool)) if amodal is not None else None
        completed_rgba, completion_meta = complete_hidden_layer(rgb, alpha, vis, hidden, inpainting_cfg or {"method": "auto"}, device=device)
        hidden_area_ratio = float(hidden.sum() / max(1, int(vis.sum()))) if hidden is not None else 0.0
        continuity = edge_continuity_score(vis, hidden) if hidden is not None else 1.0
        edge_meta = {
            "outgoing_edges": {str(k): asdict(edge) for k, edge in node.outgoing_edges.items()},
            "incoming_edges": {str(k): asdict(edge) for k, edge in node.incoming_edges.items()},
        }
        layers.append(Layer(
            len(layers),
            f"{rank:03d}_{safe_name(seg.group)}_{safe_name(seg.label)}",
            seg.label,
            seg.group,
            rank,
            node.depth_median,
            node.depth_p10,
            node.depth_p90,
            int(vis.sum()),
            bbox_from_mask(vis),
            alpha,
            rgba,
            ar,
            sr,
            vis,
            amodal,
            [seg.id],
            sorted(node.occludes),
            sorted(node.occluded_by),
            {
                "source": seg.source,
                "score": seg.score,
                "ordering_method": ordering_method,
                "ordering_score": ordering_scores.get(sid),
                "alpha": alpha_meta,
                "alpha_quality_score": alpha_meta.get("alpha_quality_score"),
                "amodal": amodal_meta,
                "hidden_area_ratio": hidden_area_ratio,
                "completion_consistency": completion_meta.get("boundary_consistency", 1.0),
                "edge_continuity_score": continuity,
                "hidden_completion": completion_meta,
                "intrinsic_residual": iid_residual,
                "depth_stats": {
                    "median": node.depth_median,
                    "p10": node.depth_p10,
                    "p90": node.depth_p90,
                    "trimmed_mean": node.depth_trimmed_mean,
                    "boundary_median": node.depth_boundary_median,
                    "variance": node.depth_variance,
                    "confidence": node.depth_confidence,
                },
                **seg.metadata,
                **edge_meta,
            },
            hidden_mask=hidden,
            completed_rgba=completed_rgba,
        ))
    return merge_compatible_layers(layers, cfg), nodes


def build_completed_background_layer(bg_rgb: np.ndarray, albedo: np.ndarray, shading: np.ndarray, rank: int, method: str) -> Layer:
    alpha = np.ones(bg_rgb.shape[:2], dtype=np.float32)
    rgba = rgba_from_rgb_alpha(bg_rgb, alpha)
    ar, sr = intrinsic_rgba(albedo, shading, alpha)
    return Layer(-1, f"{rank:03d}_background_completed", "background completed", "background", rank, 1.0, 1.0, 1.0, int(alpha.size), (0, 0, bg_rgb.shape[1], bg_rgb.shape[0]), alpha, rgba, ar, sr, alpha > 0, None, [], [], [], {"inpaint_method": method})


def grouped_layers(layers: list[Layer], bins: int = 3) -> list[Layer]:
    fg = [l for l in layers if l.group != "background"]
    bg = [l for l in layers if l.group == "background"]
    if not fg:
        return bg
    depths = np.array([l.depth_median for l in fg], dtype=np.float32)
    qs = np.quantile(depths, np.linspace(0, 1, bins + 1)) if len(depths) > 1 else np.array([0, 1], dtype=np.float32)
    buckets: dict[tuple[str, int], list[Layer]] = defaultdict(list)
    for l in fg:
        bid = int(np.searchsorted(qs[1:-1], l.depth_median, side="right"))
        buckets[(l.group, bid)].append(l)
    out: list[Layer] = []
    for (group, bid), bucket in sorted(buckets.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        comp = composite_layers_near_to_far(bucket)
        alpha = image_to_float(comp)[..., 3]
        first = sorted(bucket, key=lambda l: l.rank)[0]
        out.append(Layer(len(out), f"{len(out):03d}_{safe_name(group)}_depthbin_{bid}", f"{group} depth bin {bid}", group, len(out), float(np.median([b.depth_median for b in bucket])), float(np.min([b.depth_p10 for b in bucket])), float(np.max([b.depth_p90 for b in bucket])), int((alpha > 0.05).sum()), bbox_from_mask(alpha > 0.05), alpha, comp, first.albedo_rgba, first.shading_rgba, alpha > 0.05, None, [], [], [], {"members": [b.name for b in bucket]}))
    for b in bg:
        idx = len(out)
        group_name = safe_name(b.group)
        base_name = _base_label(b.label)
        name = f"{idx:03d}_{group_name}" if base_name == group_name else f"{idx:03d}_{group_name}_{base_name}"
        out.append(replace(b, id=idx, rank=idx, name=name))
    return out


def graph_json(layers: list[Layer], nodes: dict[int, Node]) -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    for _, n in sorted(nodes.items()):
        for _, edge in sorted(n.outgoing_edges.items()):
            edges.append(asdict(edge))
    return {
        "layers_near_to_far": [
            {
                "rank": l.rank,
                "name": l.name,
                "label": l.label,
                "group": l.group,
                "depth_median": l.depth_median,
                "depth_p10": l.depth_p10,
                "depth_p90": l.depth_p90,
                "depth_trimmed_mean": l.metadata.get("depth_stats", {}).get("trimmed_mean"),
                "depth_boundary_median": l.metadata.get("depth_stats", {}).get("boundary_median"),
                "depth_variance": l.metadata.get("depth_stats", {}).get("variance"),
                "depth_confidence": l.metadata.get("depth_stats", {}).get("confidence"),
                "area": l.area,
                "bbox": l.bbox,
                "occludes": l.occludes,
                "occluded_by": l.occluded_by,
                "source_segment_ids": l.source_segment_ids,
                "metadata": l.metadata,
            }
            for l in sorted(layers, key=lambda x: x.rank)
        ],
        "occlusion_edges": edges,
        "segment_nodes": [
            {
                "segment_id": sid,
                "label": n.segment.label,
                "group": n.segment.group,
                "depth_median": n.depth_median,
                "depth_trimmed_mean": n.depth_trimmed_mean,
                "depth_boundary_median": n.depth_boundary_median,
                "depth_variance": n.depth_variance,
                "depth_confidence": n.depth_confidence,
                "occludes": sorted(n.occludes),
                "occluded_by": sorted(n.occluded_by),
                "removed_edges": n.removed_edges,
            }
            for sid, n in sorted(nodes.items())
        ],
    }
