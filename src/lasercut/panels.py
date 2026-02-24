"""Load STEP bodies, thicken panels inward, detect shared edges."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    name: str
    solid: cq.Shape
    outer_normal: tuple[float, float, float]
    width: float   # longer in-plane dimension
    height: float  # shorter in-plane dimension
    outer_face: cq.Face | None = None
    outer_edges: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = field(
        default_factory=list
    )


@dataclass
class SharedEdge:
    panel_a: str
    panel_b: str
    edge_length: float
    start_3d: tuple[float, float, float]
    end_3d: tuple[float, float, float]


@dataclass
class BinModel:
    panels: dict[str, Panel]
    shared_edges: list[SharedEdge]
    thickness: float
    source_solid: cq.Shape | None = None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _vec_len(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _vec_sub(a: tuple, b: tuple) -> tuple:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_dot(a: tuple, b: tuple) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a: tuple, b: tuple) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _pt_dist(a: tuple, b: tuple) -> float:
    return _vec_len(_vec_sub(a, b))


def _point_to_line_dist(p: tuple, a: tuple, b: tuple) -> float:
    """Distance from point *p* to the infinite line through *a* and *b*."""
    ab = _vec_sub(b, a)
    ap = _vec_sub(p, a)
    cross = _vec_cross(ab, ap)
    ab_len = _vec_len(ab)
    if ab_len < 1e-9:
        return _pt_dist(p, a)
    return _vec_len(cross) / ab_len


def _edge_overlap_length(
    a1: tuple, a2: tuple, b1: tuple, b2: tuple
) -> float:
    """Return the overlap length of two colinear segments projected onto their shared direction."""
    direction = _vec_sub(a2, a1)
    d_len = _vec_len(direction)
    if d_len < 1e-9:
        return 0.0
    d_unit = (direction[0] / d_len, direction[1] / d_len, direction[2] / d_len)

    # Project all four points onto the line direction
    t_a1 = _vec_dot(_vec_sub(a1, a1), d_unit)  # = 0
    t_a2 = _vec_dot(_vec_sub(a2, a1), d_unit)  # = d_len
    t_b1 = _vec_dot(_vec_sub(b1, a1), d_unit)
    t_b2 = _vec_dot(_vec_sub(b2, a1), d_unit)

    seg_a = (min(t_a1, t_a2), max(t_a1, t_a2))
    seg_b = (min(t_b1, t_b2), max(t_b1, t_b2))

    overlap_start = max(seg_a[0], seg_b[0])
    overlap_end = min(seg_a[1], seg_b[1])

    if overlap_end <= overlap_start + 0.1:
        return 0.0
    return overlap_end - overlap_start


# ---------------------------------------------------------------------------
# STEP loading & face analysis
# ---------------------------------------------------------------------------

def _extract_outer_wire_edges(
    face: cq.Face,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Return edge endpoints [(p1, p2), ...] from the outer wire of a face."""
    explorer = TopExp_Explorer(face.wrapped, TopAbs_WIRE)
    if not explorer.More():
        return []
    outer_wire = TopoDS.Wire_s(explorer.Current())

    edges: list[tuple[tuple, tuple]] = []
    edge_exp = TopExp_Explorer(outer_wire, TopAbs_EDGE)
    while edge_exp.More():
        edge = TopoDS.Edge_s(edge_exp.Current())
        verts: list[tuple[float, float, float]] = []
        v_exp = TopExp_Explorer(edge, TopAbs_VERTEX)
        while v_exp.More():
            v = TopoDS.Vertex_s(v_exp.Current())
            pnt = BRep_Tool.Pnt_s(v)
            verts.append((pnt.X(), pnt.Y(), pnt.Z()))
            v_exp.Next()
        if len(verts) >= 2:
            edges.append((verts[0], verts[1]))
        edge_exp.Next()
    return edges


def _extract_panels_from_single_solid(
    obj: cq.Shape,
    centroid: tuple[float, float, float],
) -> list[dict]:
    """Extract major outward panel faces from a single closed STEP solid.

    The input solid is expected to represent the full bin shell (one body).
    We select outward planar faces with significant area and treat each as
    a panel source face.
    """
    faces = cq.Workplane("XY").add(obj).faces().vals()

    candidates: list[tuple[float, tuple[float, float, float], cq.Face]] = []
    for face in faces:
        try:
            if face.geomType() != "PLANE":
                continue
            normal = face.normalAt()
            center = face.Center()
            area = face.Area()
        except Exception:
            continue

        normal_t = (normal.x, normal.y, normal.z)
        to_center = (
            center.x - centroid[0],
            center.y - centroid[1],
            center.z - centroid[2],
        )

        # Keep only outward-oriented faces.
        if _vec_dot(normal_t, to_center) <= 0:
            continue
        candidates.append((area, normal_t, face))

    if not candidates:
        raise ValueError("No outward planar faces found in single-solid STEP")

    max_area = max(a for a, _, _ in candidates)
    area_cutoff = max(100.0, max_area * 0.08)
    major = [item for item in candidates if item[0] >= area_cutoff]

    # Fallback for unusual small models.
    if len(major) < 3:
        area_cutoff = max(50.0, max_area * 0.03)
        major = [item for item in candidates if item[0] >= area_cutoff]

    major.sort(key=lambda item: -item[0])

    bodies_data: list[dict] = []
    for idx, (_, normal_t, face) in enumerate(major):
        edges = _extract_outer_wire_edges(face)
        if not edges:
            continue
        bodies_data.append(
            {
                "idx": idx,
                "outer_face": face,
                "outer_normal": normal_t,
                "edges": edges,
            }
        )

    if not bodies_data:
        raise ValueError("No panel outlines extracted from single-solid STEP")

    return bodies_data


def _classify_panel(normal: tuple[float, float, float]) -> str:
    """Return a human-readable classification based on the outer normal."""
    nx, ny, nz = normal
    abs_ny = abs(ny)

    # End walls have normal pointing purely in +/-Y
    if abs_ny > 0.95:
        return "end_wall"

    # Bottom: normal has strong -Z component (pointing down/outward)
    # For 30-deg tilted bins the bottom normal is roughly (-0.5, 0, -0.866)
    if nz < -0.7:
        return "bottom"

    # Right/left side walls: normal has strong X component
    if abs(nx) > 0.7:
        return "side_wall"

    # Gussets: mixed normal
    return "gusset"


def _name_panels(
    bodies_data: list[dict],
) -> list[str]:
    """Assign unique names to panels based on their normals and positions."""
    names: list[str] = []
    category_counts: dict[str, int] = {}
    side_wall_total = sum(
        1 for bd in bodies_data if _classify_panel(bd["outer_normal"]) == "side_wall"
    )

    for bd in bodies_data:
        cat = _classify_panel(bd["outer_normal"])
        count = category_counts.get(cat, 0)

        if cat == "end_wall":
            # Distinguish front vs right/back by Y position.
            # For single-side-wall bins, the side wall is the functional back wall
            # (inset/lip wall), so the +Y end wall is the right wall.
            ny = bd["outer_normal"][1]
            if ny < 0:
                name = "front_wall"
            else:
                name = "right_wall" if side_wall_total == 1 else "back_wall"
        elif cat == "bottom":
            name = "bottom" if count == 0 else f"bottom_{count}"
        elif cat == "side_wall":
            if side_wall_total == 1:
                name = "back_wall"
            else:
                nx = bd["outer_normal"][0]
                name = "right_wall" if nx > 0 else "left_wall"
        elif cat == "gusset":
            ny = bd["outer_normal"][1]
            if ny > 0.5:
                name = "back_gusset" if count == 0 else f"back_gusset_{count}"
            elif ny < -0.5:
                name = "front_gusset" if count == 0 else f"front_gusset_{count}"
            else:
                name = f"gusset_{count}"
        else:
            name = f"panel_{count}"

        # Handle duplicate names
        if name in names:
            name = f"{name}_{count}"
        names.append(name)
        category_counts[cat] = count + 1

    return names


# ---------------------------------------------------------------------------
# Thickening
# ---------------------------------------------------------------------------

def _thicken_face_inward(
    outer_face: cq.Face,
    outer_normal: tuple[float, float, float],
    thickness: float,
) -> cq.Shape:
    """Extrude the outer face's wire inward by *thickness* to create a slab.

    The outer surface stays exactly in place; the inner surface moves inward.
    """
    explorer = TopExp_Explorer(outer_face.wrapped, TopAbs_WIRE)
    outer_wire = TopoDS.Wire_s(explorer.Current())

    face_for_prism = BRepBuilderAPI_MakeFace(outer_wire, True).Face()

    inward = gp_Vec(
        -outer_normal[0] * thickness,
        -outer_normal[1] * thickness,
        -outer_normal[2] * thickness,
    )
    prism = BRepPrimAPI_MakePrism(face_for_prism, inward)
    return cq.Shape(prism.Shape())


# ---------------------------------------------------------------------------
# Shared edge detection
# ---------------------------------------------------------------------------

def _find_shared_edges(
    panels: dict[str, Panel],
    tolerance: float = 5.0,
    min_edge_length: float = 10.0,
) -> list[SharedEdge]:
    """Find edges shared between panels.

    Two edges are considered shared if they are colinear (within *tolerance*)
    and have significant overlap (> *min_edge_length*).

    Handles notched/fragmented edges by aggregating colinear overlap from
    multiple short edges against a single long reference edge.
    """
    shared: list[SharedEdge] = []
    panel_names = list(panels.keys())

    for i, name_a in enumerate(panel_names):
        for j in range(i + 1, len(panel_names)):
            name_b = panel_names[j]
            best = _best_shared_edge(
                panels[name_a], panels[name_b], tolerance, min_edge_length
            )
            if best is not None:
                shared.append(best)

    return shared


def _collect_colinear_overlap(
    ref_edge: tuple[tuple, tuple],
    other_edges: list[tuple[tuple, tuple]],
    tolerance: float,
) -> tuple[float, float, float]:
    """Sum up overlap of all edges in *other_edges* colinear to *ref_edge*.

    Returns (total_overlap, t_min, t_max) where t_min/t_max are the
    parametric extents along *ref_edge*.
    """
    ref_a, ref_b = ref_edge
    direction = _vec_sub(ref_b, ref_a)
    d_len = _vec_len(direction)
    if d_len < 1e-9:
        return (0.0, 0.0, 0.0)
    d_unit = (direction[0] / d_len, direction[1] / d_len, direction[2] / d_len)

    intervals: list[tuple[float, float]] = []

    for eb in other_edges:
        # Check colinearity
        d1 = _point_to_line_dist(eb[0], ref_a, ref_b)
        d2 = _point_to_line_dist(eb[1], ref_a, ref_b)
        if d1 > tolerance or d2 > tolerance:
            continue

        # Compute overlap
        t1 = _vec_dot(_vec_sub(eb[0], ref_a), d_unit)
        t2 = _vec_dot(_vec_sub(eb[1], ref_a), d_unit)
        t_lo, t_hi = min(t1, t2), max(t1, t2)

        # Clip to ref_edge range [0, d_len]
        t_lo = max(t_lo, 0.0)
        t_hi = min(t_hi, d_len)
        if t_hi - t_lo > 0.1:
            intervals.append((t_lo, t_hi))

    if not intervals:
        return (0.0, 0.0, 0.0)

    # Merge overlapping intervals and sum total length
    intervals.sort()
    merged: list[tuple[float, float]] = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + 0.1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    total = sum(hi - lo for lo, hi in merged)
    t_min = merged[0][0]
    t_max = merged[-1][1]
    return (total, t_min, t_max)


def _best_shared_edge(
    pa: Panel,
    pb: Panel,
    tolerance: float,
    min_edge_length: float,
) -> SharedEdge | None:
    """Return the best (longest) shared edge between two panels, or None.

    Uses aggregated colinear overlap so that notched/fragmented edges
    (e.g. existing finger joints in the STEP) still match.
    """
    best_length = 0.0
    best_edge: SharedEdge | None = None

    # Try each edge in pa as the reference, accumulate overlap from all pb edges
    for ea in pa.outer_edges:
        ea_len = _pt_dist(ea[0], ea[1])
        if ea_len < 5.0:  # skip very short edges
            continue

        total, t_min, t_max = _collect_colinear_overlap(ea, pb.outer_edges, tolerance)
        if total >= min_edge_length and total > best_length:
            best_length = total
            d = _vec_sub(ea[1], ea[0])
            d_len = _vec_len(d)
            d_unit = (d[0] / d_len, d[1] / d_len, d[2] / d_len)
            start_3d = tuple(ea[0][k] + t_min * d_unit[k] for k in range(3))
            end_3d = tuple(ea[0][k] + t_max * d_unit[k] for k in range(3))
            best_edge = SharedEdge(
                panel_a=pa.name, panel_b=pb.name,
                edge_length=t_max - t_min,
                start_3d=start_3d, end_3d=end_3d,
            )

    # Also try each edge in pb as the reference
    for eb in pb.outer_edges:
        eb_len = _pt_dist(eb[0], eb[1])
        if eb_len < 5.0:
            continue

        total, t_min, t_max = _collect_colinear_overlap(eb, pa.outer_edges, tolerance)
        if total >= min_edge_length and total > best_length:
            best_length = total
            d = _vec_sub(eb[1], eb[0])
            d_len = _vec_len(d)
            d_unit = (d[0] / d_len, d[1] / d_len, d[2] / d_len)
            start_3d = tuple(eb[0][k] + t_min * d_unit[k] for k in range(3))
            end_3d = tuple(eb[0][k] + t_max * d_unit[k] for k in range(3))
            best_edge = SharedEdge(
                panel_a=pa.name, panel_b=pb.name,
                edge_length=t_max - t_min,
                start_3d=start_3d, end_3d=end_3d,
            )

    return best_edge


# ---------------------------------------------------------------------------
# In-plane dimensions
# ---------------------------------------------------------------------------

def _compute_in_plane_dims(
    edges: list[tuple[tuple, tuple]],
    normal: tuple[float, float, float],
) -> tuple[float, float]:
    """Compute width and height of a panel from its outer wire edges.

    Projects all edge endpoints onto the plane perpendicular to *normal*,
    then returns the bounding rectangle dimensions (longer = width).
    """
    if not edges:
        return (0.0, 0.0)

    # Build a local 2D coordinate system on the plane
    nx, ny, nz = normal

    # Pick an arbitrary axis not parallel to normal
    if abs(nx) < 0.9:
        ref = (1, 0, 0)
    else:
        ref = (0, 1, 0)

    # u = ref x normal (normalized)
    u = _vec_cross(ref, normal)
    u_len = _vec_len(u)
    u = (u[0] / u_len, u[1] / u_len, u[2] / u_len)

    # v = normal x u
    v = _vec_cross(normal, u)
    v_len = _vec_len(v)
    v = (v[0] / v_len, v[1] / v_len, v[2] / v_len)

    # Project all points
    us = []
    vs = []
    for p1, p2 in edges:
        for pt in [p1, p2]:
            us.append(_vec_dot(pt, u))
            vs.append(_vec_dot(pt, v))

    u_span = max(us) - min(us)
    v_span = max(vs) - min(vs)

    width = max(u_span, v_span)
    height = min(u_span, v_span)
    return (round(width, 1), round(height, 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_step_panels(step_path: str, thickness: float = 3.2) -> BinModel:
    """Load a STEP file and return a BinModel with thickened panels and shared edges.

    Parameters
    ----------
    step_path : str
        Path to the STEP file containing separate bodies for each panel.
    thickness : float
        Target panel thickness in mm. Panels are thickened inward so
        the outer surface stays fixed.
    """
    result = cq.importers.importStep(step_path)
    solids = result.objects

    if not solids:
        raise ValueError(f"No solid bodies found in {step_path}")

    # Compute bin centroid (average of all body bounding box centers)
    bb_centers = []
    for obj in solids:
        wp = cq.Workplane("XY").add(obj)
        bb = wp.val().BoundingBox()
        bb_centers.append(
            ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
        )
    centroid = tuple(
        sum(c[k] for c in bb_centers) / len(bb_centers) for k in range(3)
    )

    # Analyse geometry: either one body per panel OR a single full-shell body.
    bodies_data: list[dict] = []
    if len(solids) == 1:
        bodies_data = _extract_panels_from_single_solid(solids[0], centroid)
    else:
        for idx, obj in enumerate(solids):
            wp = cq.Workplane("XY").add(obj)
            faces = wp.faces().vals()

            # Sort faces by area (largest first)
            face_info = []
            for f in faces:
                try:
                    normal = f.normalAt()
                    face_info.append((f.Area(), (normal.x, normal.y, normal.z), f))
                except Exception:
                    pass
            face_info.sort(key=lambda x: -x[0])
            if not face_info:
                continue

            # Pick the outer face: the one whose normal points away from the centroid
            outer_face = None
            outer_normal = None
            for area, normal, face in face_info[:2]:
                center = face.Center()
                to_center = (
                    center.x - centroid[0],
                    center.y - centroid[1],
                    center.z - centroid[2],
                )
                dot = _vec_dot(normal, to_center)
                if dot > 0:
                    outer_face = face
                    outer_normal = normal
                    break

            if outer_face is None:
                # Fallback: use the largest face
                outer_face = face_info[0][2]
                outer_normal = face_info[0][1]

            edges = _extract_outer_wire_edges(outer_face)

            bodies_data.append(
                {
                    "idx": idx,
                    "outer_face": outer_face,
                    "outer_normal": outer_normal,
                    "edges": edges,
                }
            )

    if not bodies_data:
        raise ValueError(f"No panel faces could be extracted from {step_path}")

    # Assign unique names
    names = _name_panels(bodies_data)

    # Build Panel objects with thickened solids
    panels: dict[str, Panel] = {}
    for bd, name in zip(bodies_data, names):
        thickened = _thicken_face_inward(
            bd["outer_face"], bd["outer_normal"], thickness
        )
        w, h = _compute_in_plane_dims(bd["edges"], bd["outer_normal"])
        panels[name] = Panel(
            name=name,
            solid=thickened,
            outer_normal=bd["outer_normal"],
            width=w,
            height=h,
            outer_face=bd["outer_face"],
            outer_edges=bd["edges"],
        )

    # Detect shared edges
    shared_edges = _find_shared_edges(panels)

    source_solid = solids[0] if len(solids) == 1 else None
    return BinModel(
        panels=panels,
        shared_edges=shared_edges,
        thickness=thickness,
        source_solid=source_solid,
    )
