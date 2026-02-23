"""Finger joint generation via 3D boolean cuts on panel solids.

Supports two joint types:
- **Finger joints** for edge-joined panels (shared edge runs along both panels' boundaries)
- **Through-slots** for inset panels (shared edge cuts across one panel's face interior)
"""

from __future__ import annotations

import cadquery as cq
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from lasercut.panels import (
    BinModel,
    Panel,
    SharedEdge,
    _vec_sub,
    _vec_len,
    _vec_cross,
    _vec_dot,
    _pt_dist,
    _point_to_line_dist,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _vec_len(v)
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _scale(v: tuple, s: float) -> tuple:
    return (v[0] * s, v[1] * s, v[2] * s)


def _add(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _project_point_to_panel_plane(
    pt: tuple[float, float, float],
    panel: Panel,
) -> tuple[float, float, float]:
    """Project 3D point onto panel's outer-face plane along panel normal."""
    if not panel.outer_edges:
        return pt

    plane_pt = panel.outer_edges[0][0]
    n = panel.outer_normal
    v = _vec_sub(pt, plane_pt)
    d = _vec_dot(v, n)
    return (
        pt[0] - d * n[0],
        pt[1] - d * n[1],
        pt[2] - d * n[2],
    )


def _project_edge_to_panel(
    se: SharedEdge,
    panel: Panel,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Project shared-edge endpoints onto panel plane, preserving edge direction."""
    p0 = _project_point_to_panel_plane(se.start_3d, panel)
    p1 = _project_point_to_panel_plane(se.end_3d, panel)

    # Keep projected edge direction consistent with the original shared edge.
    if _vec_dot(_vec_sub(p1, p0), _vec_sub(se.end_3d, se.start_3d)) < 0:
        return (p1, p0)
    return (p0, p1)


def _panel_center(panel: Panel) -> tuple[float, float, float]:
    bb = panel.solid.BoundingBox()
    return (
        (bb.xmin + bb.xmax) / 2.0,
        (bb.ymin + bb.ymax) / 2.0,
        (bb.zmin + bb.zmax) / 2.0,
    )


def _edge_inward_direction(
    panel: Panel,
    edge_start: tuple[float, float, float],
    edge_end: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the in-plane direction that points from the shared edge into panel interior."""
    edge_dir = _normalize(_vec_sub(edge_end, edge_start))
    normal = panel.outer_normal
    candidate = _normalize(_vec_cross(normal, edge_dir))
    mid = tuple((edge_start[k] + edge_end[k]) / 2 for k in range(3))
    center = _panel_center(panel)
    to_center = _vec_sub(center, mid)

    if _vec_dot(candidate, to_center) < 0:
        return (-candidate[0], -candidate[1], -candidate[2])
    return candidate


def _to_cuttable(obj) -> cq.Shape:
    """Return a CadQuery shape that preserves full boolean results.

    Boolean cuts can produce compounds. We must keep the complete shape
    (not just the first solid) or panel geometry can be lost.
    """
    if isinstance(obj, cq.Shape):
        return obj

    wrapped = obj.wrapped
    from OCP.TopoDS import TopoDS_Solid, TopoDS_Compound
    if isinstance(wrapped, TopoDS_Solid):
        return cq.Solid(wrapped)
    if isinstance(wrapped, TopoDS_Compound):
        return cq.Compound(wrapped)
    return cq.Shape(wrapped)


def _make_oriented_box(
    origin: tuple[float, float, float],
    x_dir: tuple[float, float, float],
    y_dir: tuple[float, float, float],
    z_dir: tuple[float, float, float],
    dx: float,
    dy: float,
    dz: float,
) -> cq.Solid:
    """Create a box at *origin* with axes (x_dir, y_dir, z_dir) and dimensions (dx, dy, dz).

    BRepPrimAPI_MakeBox(gp_Ax2, dx, dy, dz) creates the box so that:
    - dx extends along the ax2 X direction
    - dy extends along the ax2 Y direction (= Z cross X)
    - dz extends along the ax2 Z direction

    We set ax2 Z = z_dir, X = x_dir, and ensure Y = Z cross X matches y_dir.
    If Z cross X gives -y_dir, we flip X so that Y comes out right.
    """
    pt = gp_Pnt(origin[0], origin[1], origin[2])

    # gp_Ax2 Y axis = Z cross X.  Check if it matches y_dir.
    computed_y = _vec_cross(z_dir, x_dir)
    if _vec_dot(computed_y, y_dir) < 0:
        # Flip: use -x_dir as X axis, negative dx won't work,
        # so instead shift origin by dx along x_dir and negate x_dir
        origin = _add(origin, _scale(x_dir, dx))
        pt = gp_Pnt(origin[0], origin[1], origin[2])
        x_dir = (-x_dir[0], -x_dir[1], -x_dir[2])

    ax2 = gp_Ax2(pt, gp_Dir(z_dir[0], z_dir[1], z_dir[2]),
                  gp_Dir(x_dir[0], x_dir[1], x_dir[2]))
    maker = BRepPrimAPI_MakeBox(ax2, dx, dy, dz)
    return cq.Solid(maker.Shape())


# ---------------------------------------------------------------------------
# Corner keepout detection
# ---------------------------------------------------------------------------

def _find_corner_points(
    shared_edges: list[SharedEdge],
    tolerance: float = 5.0,
) -> list[tuple[tuple[float, float, float], set[str]]]:
    """Find points where 3+ panels meet. Returns (point, set_of_panel_names)."""
    endpoints: list[tuple[tuple[float, float, float], str, str]] = []
    for se in shared_edges:
        endpoints.append((se.start_3d, se.panel_a, se.panel_b))
        endpoints.append((se.end_3d, se.panel_a, se.panel_b))

    groups: list[tuple[tuple[float, float, float], set[str]]] = []
    used = [False] * len(endpoints)
    for i in range(len(endpoints)):
        if used[i]:
            continue
        pt, pa, pb = endpoints[i]
        panels = {pa, pb}
        used[i] = True
        for j in range(i + 1, len(endpoints)):
            if used[j]:
                continue
            if _pt_dist(pt, endpoints[j][0]) < tolerance:
                panels.add(endpoints[j][1])
                panels.add(endpoints[j][2])
                used[j] = True
        groups.append((pt, panels))

    # Only return corners where 3+ panels meet
    return [(pt, panels) for pt, panels in groups if len(panels) >= 3]


def _corner_keepout_for_edge(
    se: SharedEdge,
    corners: list[tuple[tuple[float, float, float], set[str]]],
    thickness: float,
    tolerance: float = 5.0,
) -> tuple[float, float]:
    """Return keepout distances (start_keepout, end_keepout) for the shared edge.

    If an endpoint is near a corner where 3+ panels meet, we add a keepout
    equal to *thickness* so fingers don't interfere at the corner.
    """
    start_keepout = 0.0
    end_keepout = 0.0

    for pt, panels in corners:
        if _pt_dist(se.start_3d, pt) < tolerance:
            if se.panel_a in panels and se.panel_b in panels:
                start_keepout = thickness
        if _pt_dist(se.end_3d, pt) < tolerance:
            if se.panel_a in panels and se.panel_b in panels:
                end_keepout = thickness

    return (start_keepout, end_keepout)


# ---------------------------------------------------------------------------
# Inset detection
# ---------------------------------------------------------------------------

def _is_edge_on_boundary(
    se: SharedEdge,
    panel: Panel,
    tolerance: float = 0.6,
) -> bool:
    """Check if the shared edge runs along the panel's outer boundary.

    Returns True if enough of the shared edge overlaps colinear panel
    boundary segments (within *tolerance*).
    Returns False if the shared edge cuts through the panel's interior (inset).
    """
    se_start, se_end = _project_edge_to_panel(se, panel)
    se_vec = _vec_sub(se_end, se_start)
    se_len = _vec_len(se_vec)
    if se_len < 1e-6:
        return False
    se_dir = (se_vec[0] / se_len, se_vec[1] / se_len, se_vec[2] / se_len)

    intervals: list[tuple[float, float]] = []

    for pe in panel.outer_edges:
        pe_vec = _vec_sub(pe[1], pe[0])
        pe_len = _vec_len(pe_vec)
        if pe_len < 0.5:
            continue
        pe_dir = (pe_vec[0] / pe_len, pe_vec[1] / pe_len, pe_vec[2] / pe_len)

        # Directions must be roughly parallel
        if abs(_vec_dot(se_dir, pe_dir)) < 0.98:
            continue

        # Panel edge should lie on the same line as the shared edge.
        d0 = _point_to_line_dist(pe[0], se_start, se_end)
        d1 = _point_to_line_dist(pe[1], se_start, se_end)
        if d0 > tolerance or d1 > tolerance:
            continue

        # Project edge onto shared-edge param axis and collect overlap interval.
        t0 = _vec_dot(_vec_sub(pe[0], se_start), se_dir)
        t1 = _vec_dot(_vec_sub(pe[1], se_start), se_dir)
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        lo = max(lo, 0.0)
        hi = min(hi, se_len)
        if hi - lo > 0.05:
            intervals.append((lo, hi))

    if not intervals:
        return False

    # Merge overlapping intervals and require substantial overlap.
    intervals.sort()
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    overlap = sum(hi - lo for lo, hi in merged)
    return overlap >= 0.25 * se_len


def _classify_joint_type(
    se: SharedEdge,
    panels: dict[str, Panel],
    tolerance: float = 0.6,
) -> tuple[str, str | None]:
    """Classify a shared edge as 'finger' or 'through_slot'.

    Returns (joint_type, slot_panel_name):
    - ('finger', None) for edge-joined panels
    - ('through_slot', panel_name) where panel_name is the panel that gets the slot
      (the one where the shared edge is in the interior, not on boundary)
    """
    pa = panels[se.panel_a]
    pb = panels[se.panel_b]

    on_boundary_a = _is_edge_on_boundary(se, pa, tolerance)
    on_boundary_b = _is_edge_on_boundary(se, pb, tolerance)

    if on_boundary_a and not on_boundary_b:
        # Edge is on A's boundary but cuts through B's interior -> slot in B
        return ("through_slot", se.panel_b)
    elif on_boundary_b and not on_boundary_a:
        # Edge is on B's boundary but cuts through A's interior -> slot in A
        return ("through_slot", se.panel_a)
    else:
        # Both on boundary (normal edge-join) or both interior (ambiguous -> finger)
        return ("finger", None)


# ---------------------------------------------------------------------------
# Finger layout
# ---------------------------------------------------------------------------

def _compute_finger_layout(
    edge_length: float,
    finger_width: float,
    start_keepout: float,
    end_keepout: float,
) -> list[tuple[float, float]]:
    """Compute finger positions along the edge.

    Returns a list of (offset, width) for each finger.
    Fingers start and end with a tab (odd count).
    The usable range is [start_keepout, edge_length - end_keepout].
    """
    usable_start = start_keepout
    usable_length = edge_length - start_keepout - end_keepout

    if usable_length <= 0:
        return []

    # Compute number of fingers: odd count, each ~finger_width
    n = max(1, round(usable_length / finger_width))
    if n % 2 == 0:
        n += 1  # ensure odd

    actual_width = usable_length / n

    fingers = []
    for i in range(n):
        offset = usable_start + i * actual_width
        fingers.append((offset, actual_width))

    return fingers


# ---------------------------------------------------------------------------
# Main joint application
# ---------------------------------------------------------------------------

def _apply_through_slot(
    se: SharedEdge,
    slot_panel_name: str,
    panels: dict[str, Panel],
    corners: list[tuple[tuple[float, float, float], set[str]]],
    thickness: float,
) -> None:
    """Cut a through-slot in one panel and matching tabs on the other.

    The *slot_panel_name* panel gets a continuous rectangular slot cut through it.
    The other panel gets trimmed so only tabs protrude through the slot.

    Parameters
    ----------
    se : SharedEdge
        The shared edge between the two panels.
    slot_panel_name : str
        Name of the panel that receives the through-slot (the one the other
        panel is inset into).
    panels : dict
        Mutable dict of panels -- solids are modified in place.
    corners : list
        Corner points from ``_find_corner_points``.
    thickness : float
        Material thickness in mm.
    """
    # Identify which panel is slot vs wall (the one sliding through)
    if slot_panel_name == se.panel_a:
        slot_panel = panels[se.panel_a]
    else:
        slot_panel = panels[se.panel_b]

    # Edge direction in slot-panel plane
    slot_start, slot_end = _project_edge_to_panel(se, slot_panel)
    d = _vec_sub(slot_end, slot_start)
    d_len = _vec_len(d)
    if d_len < 1e-6:
        return
    edge_dir = _normalize(d)

    # Keepouts at corners
    start_keepout, end_keepout = _corner_keepout_for_edge(se, corners, thickness)

    slot_length = d_len - start_keepout - end_keepout
    if slot_length <= 0:
        return

    # Direction through the slot panel thickness (perpendicular to slot panel face)
    into_slot = _normalize((
        -slot_panel.outer_normal[0],
        -slot_panel.outer_normal[1],
        -slot_panel.outer_normal[2],
    ))

    # In-plane direction from shared edge into slot panel interior.
    slot_in_plane = _edge_inward_direction(slot_panel, slot_start, slot_end)

    # The slot origin is at projected edge start + keepout along edge_dir
    slot_origin = _add(slot_start, _scale(edge_dir, start_keepout))

    # Cut through-slot in the slot panel:
    # A box that extends along edge_dir for slot_length,
    # extends into slot-panel interior for wall thickness,
    # and extends through the slot panel's full thickness
    # We overshoot the slot depth by 2x thickness to ensure it goes all the way through
    solid_slot = _to_cuttable(slot_panel.solid)
    slot_box = _make_oriented_box(
        origin=slot_origin,
        x_dir=edge_dir,
        y_dir=slot_in_plane,
        z_dir=into_slot,
        dx=slot_length,
        dy=thickness,
        dz=thickness * 2,  # overshoot to ensure full penetration
    )
    result = solid_slot.cut(slot_box)
    slot_panel.solid = _to_cuttable(result)


def apply_finger_joints(model: BinModel, finger_width: float = 20.0) -> BinModel:
    """Apply finger joints at all shared edges by boolean-cutting panel solids.

    For edge-joined panels (normal case):
    - Panel A gets tabs (fingers protruding), Panel B gets slots.
    - At even-indexed positions (0, 2, 4, ...): Panel A keeps material, Panel B gets cut
    - At odd-indexed positions (1, 3, 5, ...): Panel A gets cut, Panel B keeps material

    For inset panels (shared edge crosses one panel's face):
    - The panel whose face is crossed gets a continuous through-slot
    - The inset wall panel slides through that slot

    Returns a new BinModel with modified panel solids.
    """
    thickness = model.thickness

    # Copy panels so we can modify solids
    panels = {}
    for name, p in model.panels.items():
        panels[name] = Panel(
            name=p.name,
            solid=p.solid,
            outer_normal=p.outer_normal,
            width=p.width,
            height=p.height,
            outer_edges=list(p.outer_edges),
        )

    corners = _find_corner_points(model.shared_edges)

    for se in model.shared_edges:
        # Classify the joint type
        joint_type, slot_panel_name = _classify_joint_type(se, panels)

        if joint_type == "through_slot" and slot_panel_name is not None:
            print(f"  Through-slot: {se.panel_a}--{se.panel_b} (slot in {slot_panel_name})")
            _apply_through_slot(se, slot_panel_name, panels, corners, thickness)
            continue

        # Standard finger joint
        pa = panels[se.panel_a]
        pb = panels[se.panel_b]

        # Shared edge mapped onto each panel plane.
        edge_a_start, edge_a_end = _project_edge_to_panel(se, pa)
        edge_b_start, edge_b_end = _project_edge_to_panel(se, pb)

        len_a = _vec_len(_vec_sub(edge_a_end, edge_a_start))
        len_b = _vec_len(_vec_sub(edge_b_end, edge_b_start))
        edge_length = min(se.edge_length, len_a, len_b)
        if edge_length < 1e-6:
            continue
        edge_dir_a = _normalize(_vec_sub(edge_a_end, edge_a_start))
        edge_dir_b = _normalize(_vec_sub(edge_b_end, edge_b_start))

        # Direction into panel B (opposite of B's outer normal)
        into_b = _normalize((-pb.outer_normal[0], -pb.outer_normal[1], -pb.outer_normal[2]))

        # Direction into panel A (opposite of A's outer normal) -- depth direction
        into_a = _normalize((-pa.outer_normal[0], -pa.outer_normal[1], -pa.outer_normal[2]))

        # In-plane edge-to-interior direction for each panel.
        in_plane_a = _edge_inward_direction(pa, edge_a_start, edge_a_end)
        in_plane_b = _edge_inward_direction(pb, edge_b_start, edge_b_end)

        # Keepouts at corners
        start_keepout, end_keepout = _corner_keepout_for_edge(
            se, corners, thickness
        )

        # Compute finger layout
        fingers = _compute_finger_layout(edge_length, finger_width, start_keepout, end_keepout)

        # Convert current solids to cq.Solid for boolean ops
        solid_a = _to_cuttable(pa.solid)
        solid_b = _to_cuttable(pb.solid)

        for idx, (offset, fw) in enumerate(fingers):
            # Finger origins along each panel's projected edge.
            finger_origin_a = _add(edge_a_start, _scale(edge_dir_a, offset))
            finger_origin_b = _add(edge_b_start, _scale(edge_dir_b, offset))

            if idx % 2 == 0:
                # Even: Panel A keeps finger, Panel B gets slot cut
                # Cut into panel B: a box that removes material from B
                # The box sits at the shared edge and extends into B
                # Origin should be shifted to start at B's outer surface
                box = _make_oriented_box(
                    origin=finger_origin_b,
                    x_dir=edge_dir_b,
                    y_dir=into_b,
                    z_dir=in_plane_b,
                    dx=fw,
                    dy=thickness,
                    dz=thickness,
                )
                result = solid_b.cut(box)
                solid_b = _to_cuttable(result)
            else:
                # Odd: Panel B keeps finger, Panel A gets slot cut
                # Cut into panel A
                box = _make_oriented_box(
                    origin=finger_origin_a,
                    x_dir=edge_dir_a,
                    y_dir=into_a,
                    z_dir=in_plane_a,
                    dx=fw,
                    dy=thickness,
                    dz=thickness,
                )
                result = solid_a.cut(box)
                solid_a = _to_cuttable(result)

        pa.solid = solid_a
        pb.solid = solid_b

    return BinModel(panels=panels, shared_edges=model.shared_edges, thickness=thickness)
