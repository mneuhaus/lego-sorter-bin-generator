"""Project 3D planar faces to 2D outlines."""

import math
from dataclasses import dataclass
from .step_loader import PlanarFace, EdgeData


@dataclass
class Projection2D:
    """A face projected onto its local 2D coordinate system."""
    face_id: int
    label: str
    outer_polygon: list[tuple[float, float]]
    inner_polygons: list[list[tuple[float, float]]]
    # Transformation info (for mapping edges back)
    origin_3d: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    normal: tuple[float, float, float]
    # Edges in 2D (corresponding to outer_wire_edges)
    outer_edges_2d: list[tuple[tuple[float, float], tuple[float, float]]]
    # Map from 2D edge index to original 3D edge
    edge_map_3d: list[EdgeData]


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag < 1e-12:
        return (0, 0, 0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _cross(a: tuple, b: tuple) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple, b: tuple) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: tuple, b: tuple) -> tuple:
    return tuple(x - y for x, y in zip(a, b))


def _project_point(point_3d: tuple, origin: tuple, u: tuple, v: tuple) -> tuple[float, float]:
    """Project a 3D point onto a local 2D coordinate system."""
    d = _sub(point_3d, origin)
    return (_dot(d, u), _dot(d, v))


def _ordered_vertices_from_edges(edges: list[EdgeData], tol: float = 0.5) -> list[tuple]:
    """Order edge endpoints to form a continuous polygon.

    Merges very short edges (< merge_tol) into their neighbors to clean up
    fillet/chamfer remnants from 3D models.
    """
    if not edges:
        return []

    # Build ordered chain
    remaining = list(range(len(edges)))
    chain = [edges[remaining.pop(0)]]

    while remaining:
        current_end = chain[-1].end
        found = False
        # Find best match by distance
        best_idx = None
        best_dist = float('inf')
        best_reversed = False
        for idx in remaining:
            e = edges[idx]
            d_start = sum((a - b)**2 for a, b in zip(e.start, current_end))**0.5
            d_end = sum((a - b)**2 for a, b in zip(e.end, current_end))**0.5
            if d_start < best_dist and d_start <= tol:
                best_dist = d_start
                best_idx = idx
                best_reversed = False
            if d_end < best_dist and d_end <= tol:
                best_dist = d_end
                best_idx = idx
                best_reversed = True

        if best_idx is not None:
            e = edges[best_idx]
            if best_reversed:
                chain.append(EdgeData(start=e.end, end=e.start, midpoint=e.midpoint))
            else:
                chain.append(e)
            remaining.remove(best_idx)
        else:
            break

    return [e.start for e in chain]


def project_face(face: PlanarFace, label: str = "") -> Projection2D:
    """Project a planar face to 2D.

    Computes a local coordinate system on the face plane and projects
    all vertices onto it.
    """
    normal = _normalize(face.normal)

    # Choose U axis from the first edge direction
    if face.outer_wire_edges:
        e0 = face.outer_wire_edges[0]
        u_raw = _sub(e0.end, e0.start)
    else:
        # Fallback: pick an arbitrary direction perpendicular to normal
        if abs(normal[2]) < 0.9:
            u_raw = _cross(normal, (0, 0, 1))
        else:
            u_raw = _cross(normal, (1, 0, 0))

    u_axis = _normalize(u_raw)
    v_axis = _normalize(_cross(normal, u_axis))

    # Origin = first vertex of outer wire or face center
    if face.outer_wire_edges:
        origin = face.outer_wire_edges[0].start
    else:
        origin = face.center

    # Project outer wire vertices
    outer_verts_3d = _ordered_vertices_from_edges(face.outer_wire_edges)
    outer_polygon = [_project_point(v, origin, u_axis, v_axis) for v in outer_verts_3d]

    # Project outer edges (preserving edge correspondence)
    outer_edges_2d = []
    edge_map_3d = []
    for edge in face.outer_wire_edges:
        p1_2d = _project_point(edge.start, origin, u_axis, v_axis)
        p2_2d = _project_point(edge.end, origin, u_axis, v_axis)
        outer_edges_2d.append((p1_2d, p2_2d))
        edge_map_3d.append(edge)

    # Project inner wires (holes)
    inner_polygons = []
    for inner_edges in face.inner_wires_edges:
        verts_3d = _ordered_vertices_from_edges(inner_edges)
        inner_polygons.append([_project_point(v, origin, u_axis, v_axis) for v in verts_3d])

    return Projection2D(
        face_id=face.face_id,
        label=label,
        outer_polygon=outer_polygon,
        inner_polygons=inner_polygons,
        origin_3d=origin,
        u_axis=u_axis,
        v_axis=v_axis,
        normal=normal,
        outer_edges_2d=outer_edges_2d,
        edge_map_3d=edge_map_3d,
    )
