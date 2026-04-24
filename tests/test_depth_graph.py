from __future__ import annotations

import numpy as np

from layerforge.depth import orient_depth_near_to_far
from layerforge.graph import GraphEdge, build_nodes, edge_evidence, topo_order
from layerforge.segment import make_segment


def _mask(shape: tuple[int, int], y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[y0:y1, x0:x1] = True
    return out


def test_depth_orientation_auto_inverts_larger_near_depth() -> None:
    depth = np.tile(np.linspace(0.1, 0.9, 12, dtype=np.float32)[:, None], (1, 12))

    oriented, metadata = orient_depth_near_to_far(depth, {"orientation": "auto"})

    assert metadata["near_is_smaller"] is False
    assert metadata["inverted"] is True
    assert float(oriented[-1].mean()) < float(oriented[0].mean())


def test_depth_graph_orders_simple_occlusion_and_logs_evidence() -> None:
    shape = (24, 24)
    near = make_segment(0, "person", _mask(shape, 4, 4, 16, 12), 1.0, "synthetic")
    far = make_segment(1, "building", _mask(shape, 4, 13, 16, 21), 1.0, "synthetic")
    depth = np.full(shape, 0.8, dtype=np.float32)
    depth[near.mask] = 0.2
    depth[far.mask] = 0.7

    nodes = build_nodes([near, far], depth, {"occlusion_boundary_width": 2, "occlusion_depth_threshold": 0.05, "min_shared_boundary_px": 4})

    order = topo_order(nodes)
    edge = nodes[0].outgoing_edges[1]
    evidence = edge_evidence(edge)
    assert order == [0, 1]
    assert edge.relation == "in_front_of"
    assert evidence["boundary_depth_delta"] > 0.4
    assert edge.contact_score is not None


def test_depth_graph_records_same_plane_relation_without_order_constraint() -> None:
    shape = (20, 20)
    a = make_segment(0, "wall", _mask(shape, 4, 2, 14, 9), 1.0, "synthetic")
    b = make_segment(1, "window", _mask(shape, 4, 10, 14, 17), 1.0, "synthetic")
    depth = np.full(shape, 0.5, dtype=np.float32)
    depth[b.mask] = 0.51

    nodes = build_nodes([a, b], depth, {"occlusion_boundary_width": 2, "occlusion_depth_threshold": 0.05, "same_plane_depth_threshold": 0.02, "min_shared_boundary_px": 4})

    assert nodes[0].outgoing_edges[1].relation == "same_plane"
    assert nodes[0].occludes == set()
    assert topo_order(nodes) == [0, 1]


def test_topo_order_resolves_cycles_by_removing_weakest_edge() -> None:
    shape = (12, 12)
    segs = [
        make_segment(0, "a", _mask(shape, 1, 1, 5, 5), 1.0, "synthetic"),
        make_segment(1, "b", _mask(shape, 1, 6, 5, 10), 1.0, "synthetic"),
        make_segment(2, "c", _mask(shape, 6, 1, 10, 5), 1.0, "synthetic"),
    ]
    nodes = build_nodes(segs, np.zeros(shape, dtype=np.float32), {"occlusion_boundary_width": 0})
    edges = [
        GraphEdge(0, 1, 0.9, 10, 0.2, 0.1, 0.3),
        GraphEdge(1, 2, 0.1, 10, 0.2, 0.1, 0.3),
        GraphEdge(2, 0, 0.8, 10, 0.2, 0.1, 0.3),
    ]
    for edge in edges:
        nodes[edge.near_id].occludes.add(edge.far_id)
        nodes[edge.far_id].occluded_by.add(edge.near_id)
        nodes[edge.near_id].outgoing_edges[edge.far_id] = edge
        nodes[edge.far_id].incoming_edges[edge.near_id] = edge

    order = topo_order(nodes)

    assert sorted(order) == [0, 1, 2]
    assert nodes[1].removed_edges[0]["removed_reason"] == "cycle_resolution_weakest_edge"
