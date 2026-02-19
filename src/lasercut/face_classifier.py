"""Classify faces into bottom/walls and build adjacency graph."""

import math
from dataclasses import dataclass
from .step_loader import PlanarFace, EdgeData


@dataclass
class SharedEdge:
    """A shared edge between two faces."""
    face_a_id: int
    face_b_id: int
    edge_a: EdgeData
    edge_b: EdgeData
    midpoint: tuple[float, float, float]
    length: float


def _edge_length(edge: EdgeData) -> float:
    dx = edge.end[0] - edge.start[0]
    dy = edge.end[1] - edge.start[1]
    dz = edge.end[2] - edge.start[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _points_close(p1: tuple, p2: tuple, tol: float) -> bool:
    return all(abs(a - b) < tol for a, b in zip(p1, p2))


def _edges_coincident(e1: EdgeData, e2: EdgeData, tol: float = 0.1) -> bool:
    """Check if two edges are geometrically coincident (same or reversed direction)."""
    if _points_close(e1.start, e2.start, tol) and _points_close(e1.end, e2.end, tol):
        return True
    if _points_close(e1.start, e2.end, tol) and _points_close(e1.end, e2.start, tol):
        return True
    if _points_close(e1.midpoint, e2.midpoint, tol):
        length1 = _edge_length(e1)
        length2 = _edge_length(e2)
        if abs(length1 - length2) < tol:
            return True
    return False


def _dot3(a: tuple, b: tuple) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def find_shared_edges(faces: list[PlanarFace], tol: float = 0.1) -> list[SharedEdge]:
    """Find edges shared between pairs of faces."""
    shared = []
    for i, face_a in enumerate(faces):
        for j, face_b in enumerate(faces):
            if j <= i:
                continue
            for ea in face_a.outer_wire_edges:
                for eb in face_b.outer_wire_edges:
                    if _edges_coincident(ea, eb, tol):
                        length = _edge_length(ea)
                        if length > tol:
                            shared.append(SharedEdge(
                                face_a_id=face_a.face_id,
                                face_b_id=face_b.face_id,
                                edge_a=ea,
                                edge_b=eb,
                                midpoint=ea.midpoint,
                                length=length,
                            ))
    return shared


def build_adjacency(faces: list[PlanarFace], shared_edges: list[SharedEdge]) -> dict[int, list[tuple[int, SharedEdge]]]:
    """Build adjacency graph: face_id -> [(neighbor_id, shared_edge), ...]."""
    adj: dict[int, list[tuple[int, SharedEdge]]] = {f.face_id: [] for f in faces}
    for se in shared_edges:
        adj[se.face_a_id].append((se.face_b_id, se))
        adj[se.face_b_id].append((se.face_a_id, se))
    return adj


def _find_opposite_pairs(faces: list[PlanarFace], tol: float = 0.05) -> dict[int, int]:
    """Find pairs of faces with opposite normals and similar area (inner/outer of same panel).

    Returns mapping: face_id -> paired_face_id (only for the LARGER face of each pair).
    """
    pairs = {}
    used = set()

    # Sort by area descending to prefer pairing larger faces first
    sorted_faces = sorted(faces, key=lambda f: f.area, reverse=True)

    for i, fa in enumerate(sorted_faces):
        if fa.face_id in used:
            continue
        for j, fb in enumerate(sorted_faces):
            if j <= i or fb.face_id in used:
                continue
            # Check if normals are opposite
            dot = _dot3(fa.normal, fb.normal)
            if dot < -(1.0 - tol):
                # Normals are opposite; check area similarity (within 20%)
                area_ratio = min(fa.area, fb.area) / max(fa.area, fb.area)
                if area_ratio > 0.8:
                    # fa is larger (sorted by area desc), fb is smaller (inner face)
                    pairs[fa.face_id] = fb.face_id
                    used.add(fa.face_id)
                    used.add(fb.face_id)
                    break

    return pairs


def classify_faces(faces: list[PlanarFace], shared_edges: list[SharedEdge]) -> dict:
    """Classify faces into bottom plate and walls.

    Strategy:
    1. Find inner/outer face pairs (opposite normals, similar area)
    2. Take the larger (outer) face from each pair as a structural panel
    3. The largest structural panel is the bottom
    4. All other structural panels connected to the bottom (directly or via
       other panels) are walls

    Returns:
        Dictionary with keys:
        - 'bottom': PlanarFace (largest structural face)
        - 'walls': list of PlanarFace
        - 'other': list of PlanarFace
        - 'adjacency': adjacency graph
    """
    if not faces:
        return {'bottom': None, 'walls': [], 'other': [], 'adjacency': {}}

    adjacency = build_adjacency(faces, shared_edges)
    face_map = {f.face_id: f for f in faces}

    # Find inner/outer pairs
    pairs = _find_opposite_pairs(faces)
    outer_ids = set(pairs.keys())
    inner_ids = set(pairs.values())

    # Structural faces: outer faces from pairs + unpaired large faces
    structural_ids = set(outer_ids)
    for f in faces:
        if f.face_id not in outer_ids and f.face_id not in inner_ids:
            # Unpaired face — include if large enough (>5% of largest face area)
            max_area = max(ff.area for ff in faces)
            if f.area > max_area * 0.05:
                structural_ids.add(f.face_id)

    if not structural_ids:
        structural_ids = {f.face_id for f in faces}

    # Bottom: largest structural face
    bottom = max((face_map[fid] for fid in structural_ids), key=lambda f: f.area)

    # Find all walls: flood-fill from bottom through adjacency,
    # but only visiting other structural faces
    wall_ids = set()
    queue = [bottom.face_id]
    visited = {bottom.face_id}

    while queue:
        current = queue.pop(0)
        for neighbor_id, _ in adjacency.get(current, []):
            if neighbor_id in visited:
                continue
            if neighbor_id in structural_ids:
                wall_ids.add(neighbor_id)
                visited.add(neighbor_id)
                queue.append(neighbor_id)

    walls = [face_map[fid] for fid in wall_ids]
    classified_ids = {bottom.face_id} | wall_ids
    other = [f for f in faces if f.face_id not in classified_ids]

    return {
        'bottom': bottom,
        'walls': walls,
        'other': other,
        'adjacency': adjacency,
    }
