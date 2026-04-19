from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .alpha import estimate_alpha
from .compose import composite_layers_near_to_far, rgba_from_rgb_alpha
from .intrinsics import intrinsic_rgba
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
    reason: str = "boundary_local_depth"


@dataclass(slots=True)
class Node:
    segment: Segment
    depth_median: float
    depth_p10: float
    depth_p90: float
    border_touch: bool
    occludes: set[int] = field(default_factory=set)
    occluded_by: set[int] = field(default_factory=set)
    outgoing_edges: dict[int, GraphEdge] = field(default_factory=dict)
    incoming_edges: dict[int, GraphEdge] = field(default_factory=dict)


def depth_stats(mask: np.ndarray, depth: np.ndarray) -> tuple[float, float, float]:
    vals = depth[mask.astype(bool)]
    if vals.size == 0:
        return 1.0, 1.0, 1.0
    return float(np.median(vals)), float(np.percentile(vals, 10)), float(np.percentile(vals, 90))


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


def build_nodes(segments: list[Segment], depth: np.ndarray, cfg: dict[str, Any]) -> dict[int, Node]:
    nodes: dict[int, Node] = {}
    for seg in segments:
        med, p10, p90 = depth_stats(seg.mask, depth)
        nodes[seg.id] = Node(seg, med, p10, p90, touches_border(seg.mask, 3))

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
            if gap < threshold:
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
    graph[src].discard(dst)
    nodes[src].occludes.discard(dst)
    nodes[src].outgoing_edges.pop(dst, None)
    nodes[dst].occluded_by.discard(src)
    nodes[dst].incoming_edges.pop(src, None)
    return True


def topo_order(nodes: dict[int, Node]) -> list[int]:
    if not nodes:
        return []
    graph = {sid: set(node.occludes) & set(nodes) for sid, node in nodes.items()}
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
    return (fill | (expanded & hull)).astype(bool)


def visible_masks_by_order(segments: list[Segment], order: list[int]) -> dict[int, np.ndarray]:
    by_id = {s.id: s for s in segments}
    occupied = np.zeros_like(segments[0].mask, dtype=bool) if segments else np.zeros((1, 1), dtype=bool)
    visible: dict[int, np.ndarray] = {}
    for sid in order:
        m = by_id[sid].mask & ~occupied
        visible[sid] = m
        occupied |= m
    return visible


def build_layers(rgb: np.ndarray, segments: list[Segment], depth: np.ndarray, albedo: np.ndarray, shading: np.ndarray, cfg: dict[str, Any], matting_cfg: dict[str, Any]) -> tuple[list[Layer], dict[int, Node]]:
    h, w = depth.shape
    min_area = max(12, int(h * w * float(cfg.get("min_layer_area_ratio", 0.0015))))
    segments = [s for s in segments if s.area >= min_area]
    segments = split_stuff_by_depth(segments, depth, int(cfg.get("split_stuff_depth_bins", 3)), min_area)
    if len(segments) > int(cfg.get("max_layers", 64)):
        segments = sorted(segments, key=lambda s: s.area, reverse=True)[: int(cfg.get("max_layers", 64))]
        for i, s in enumerate(segments):
            s.id = i
    nodes = build_nodes(segments, depth, cfg)
    order = topo_order(nodes)
    visible = visible_masks_by_order(segments, order)
    layers: list[Layer] = []
    for rank, sid in enumerate(order):
        seg = next(s for s in segments if s.id == sid)
        vis = visible.get(sid, seg.mask)
        if int(vis.sum()) < min_area:
            continue
        node = nodes[sid]
        alpha = estimate_alpha(rgb, vis, depth, matting_cfg)
        rgba = rgba_from_rgb_alpha(rgb, alpha)
        ar, sr = intrinsic_rgba(albedo, shading, alpha)
        amodal = amodal_complete(vis, int(cfg.get("amodal_expand_px", 16))) if bool(cfg.get("amodal_enabled", True)) else None
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
            {"source": seg.source, "score": seg.score, **seg.metadata, **edge_meta},
        ))
    return layers, nodes


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
        b.rank = len(out)
        out.append(b)
    return out


def graph_json(layers: list[Layer], nodes: dict[int, Node]) -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    for _, n in sorted(nodes.items()):
        for _, edge in sorted(n.outgoing_edges.items()):
            edges.append(asdict(edge))
    return {
        "layers_near_to_far": [
            {"rank": l.rank, "name": l.name, "label": l.label, "group": l.group, "depth_median": l.depth_median, "area": l.area, "bbox": l.bbox, "occludes": l.occludes, "occluded_by": l.occluded_by, "metadata": l.metadata}
            for l in sorted(layers, key=lambda x: x.rank)
        ],
        "occlusion_edges": edges,
        "segment_nodes": [
            {"segment_id": sid, "label": n.segment.label, "group": n.segment.group, "depth_median": n.depth_median, "occludes": sorted(n.occludes), "occluded_by": sorted(n.occluded_by)}
            for sid, n in sorted(nodes.items())
        ],
    }
