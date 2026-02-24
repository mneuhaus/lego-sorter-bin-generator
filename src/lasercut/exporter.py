"""Project 3D panel faces to 2D outlines and export as SVG.

Panels are arranged in an "unfolded" layout -- as if the 3D box were opened
flat.  Adjacent panels are placed next to their shared edges with a 4 mm gap
so that finger-joint alignment can be visually verified.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepTools import BRepTools_WireExplorer

from lasercut.panels import BinModel, SharedEdge, _vec_cross, _vec_len, _vec_dot

try:
    import svgwrite
except ImportError:
    svgwrite = None  # type: ignore[assignment]

try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 2D helpers
# ---------------------------------------------------------------------------

def _rotate_pt(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (x * c - y * s, x * s + y * c)


def _rotate_pts(
    pts: list[tuple[float, float]], angle: float,
) -> list[tuple[float, float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


def _translate_pts(
    pts: list[tuple[float, float]], dx: float, dy: float,
) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in pts]


def _point_in_polygon(
    pt: tuple[float, float],
    poly: list[tuple[float, float]],
) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = pt
    inside = False
    n = len(poly)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)):
            x_inter = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_inter:
                inside = not inside
        j = i
    return inside


def _overlap_area(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> float:
    if Polygon is None:
        return 0.0
    try:
        pa = Polygon(a)
        pb = Polygon(b)
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        return pa.intersection(pb).area
    except Exception:
        return 0.0


def _distance_between(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> float:
    if Polygon is None:
        return float("inf")
    try:
        pa = Polygon(a)
        pb = Polygon(b)
        if not pa.is_valid or not pb.is_valid:
            return float("inf")
        return pa.distance(pb)
    except Exception:
        return float("inf")


def _any_overlap(
    outline: list[tuple[float, float]],
    others: list[list[tuple[float, float]]],
    min_area: float = 0.01,
) -> bool:
    for other in others:
        if _overlap_area(outline, other) > min_area:
            return True
    return False


def _total_overlap_area(
    outline: list[tuple[float, float]],
    others: list[list[tuple[float, float]]],
) -> float:
    return sum(_overlap_area(outline, other) for other in others)


def _too_close(
    outline: list[tuple[float, float]],
    others: list[list[tuple[float, float]]],
    min_clearance: float,
) -> bool:
    if Polygon is None:
        return False
    for other in others:
        if _distance_between(outline, other) < min_clearance:
            return True
    return False


def _push_out_until_clear(
    outline: list[tuple[float, float]],
    out_n: tuple[float, float],
    placed_others: list[list[tuple[float, float]]],
    step: float,
    min_clearance: float = 0.0,
    max_extra: float = 80.0,
) -> tuple[list[tuple[float, float]], float]:
    """Translate outline along out_n until it no longer overlaps placed panels."""
    needs_move = _any_overlap(outline, placed_others)
    if min_clearance > 0.0:
        needs_move = needs_move or _too_close(outline, placed_others, min_clearance)
    if not placed_others or not needs_move:
        return outline, 0.0

    def _push_dir(sign: float) -> tuple[list[tuple[float, float]], float, bool]:
        moved = list(outline)
        pushed = 0.0
        while pushed < max_extra:
            overlap = _any_overlap(moved, placed_others)
            close = min_clearance > 0.0 and _too_close(moved, placed_others, min_clearance)
            if not overlap and not close:
                break
            moved = _translate_pts(moved, out_n[0] * step * sign, out_n[1] * step * sign)
            pushed += step
        overlap = _any_overlap(moved, placed_others)
        close = min_clearance > 0.0 and _too_close(moved, placed_others, min_clearance)
        return moved, pushed * sign, not overlap and not close

    moved_fwd, pushed_fwd, ok_fwd = _push_dir(1.0)
    if ok_fwd:
        return moved_fwd, pushed_fwd

    moved_rev, pushed_rev, ok_rev = _push_dir(-1.0)
    if ok_rev:
        return moved_rev, pushed_rev

    # Neither direction fully cleared overlap; keep the better one.
    area_fwd = _total_overlap_area(moved_fwd, placed_others)
    area_rev = _total_overlap_area(moved_rev, placed_others)
    if area_rev < area_fwd:
        return moved_rev, pushed_rev
    return moved_fwd, pushed_fwd


def _bbox_area(pts: list[tuple[float, float]]) -> float:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _min_bbox_angle(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    angles: set[float] = {0.0}
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        if dx * dx + dy * dy < 1e-6:
            continue
        a = math.atan2(dy, dx) % (math.pi / 2)
        angles.add(a)
    best_angle = 0.0
    best_area = float("inf")
    for a in angles:
        rotated = _rotate_pts(pts, -a)
        area = _bbox_area(rotated)
        if area < best_area:
            best_area = area
            best_angle = a
    return best_angle


def _reflect_across_line(
    x: float, y: float, angle: float,
) -> tuple[float, float]:
    """Reflect point (x, y) across a line through the origin at *angle*."""
    c2 = math.cos(2 * angle)
    s2 = math.sin(2 * angle)
    return (c2 * x + s2 * y, s2 * x - c2 * y)


# ---------------------------------------------------------------------------
# Affine transform helper (rotation/reflection + translation)
# ---------------------------------------------------------------------------

@dataclass
class Affine2D:
    """2D affine transform: P' = M @ P + t.

    M is stored as (a, b, c, d) where M = [[a, b], [c, d]].
    """
    a: float
    b: float
    c: float
    d: float
    tx: float
    ty: float

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (self.a * x + self.b * y + self.tx,
                self.c * x + self.d * y + self.ty)

    def apply_pts(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [self.apply(x, y) for x, y in pts]

    @staticmethod
    def identity() -> "Affine2D":
        return Affine2D(1, 0, 0, 1, 0, 0)

    @staticmethod
    def from_rotation(angle: float) -> "Affine2D":
        c, s = math.cos(angle), math.sin(angle)
        return Affine2D(c, -s, s, c, 0, 0)

    @staticmethod
    def from_reflection(axis_angle: float) -> "Affine2D":
        """Reflection across a line through origin at axis_angle."""
        c2 = math.cos(2 * axis_angle)
        s2 = math.sin(2 * axis_angle)
        return Affine2D(c2, s2, s2, -c2, 0, 0)

    @staticmethod
    def from_translation(tx: float, ty: float) -> "Affine2D":
        return Affine2D(1, 0, 0, 1, tx, ty)

    def compose(self, other: "Affine2D") -> "Affine2D":
        """Return self(other(P)) = self.M @ (other.M @ P + other.t) + self.t."""
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d
        tx = self.a * other.tx + self.b * other.ty + self.tx
        ty = self.c * other.tx + self.d * other.ty + self.ty
        return Affine2D(a, b, c, d, tx, ty)


# ---------------------------------------------------------------------------
# Panel2D projection with 3D -> 2D mapping
# ---------------------------------------------------------------------------

@dataclass
class Panel2D:
    name: str
    outline: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]]
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    offset_x: float
    offset_y: float

    def project_3d(self, pt: tuple[float, float, float]) -> tuple[float, float]:
        """Map an arbitrary 3D point into this panel's local 2D coordinates."""
        return (
            _vec_dot(pt, self.u_axis) - self.offset_x,
            _vec_dot(pt, self.v_axis) - self.offset_y,
        )


def _build_local_axes(
    normal: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    nx, ny, nz = normal
    ref = (1.0, 0.0, 0.0) if abs(nx) < 0.9 else (0.0, 1.0, 0.0)
    u = _vec_cross(ref, normal)
    u_len = _vec_len(u)
    u = (u[0] / u_len, u[1] / u_len, u[2] / u_len)
    v = _vec_cross(normal, u)
    v_len = _vec_len(v)
    v = (v[0] / v_len, v[1] / v_len, v[2] / v_len)
    return u, v


def _wire_vertices_3d(wire: cq.Wire) -> list[tuple[float, float, float]]:
    explorer = BRepTools_WireExplorer(wire.wrapped)
    verts_3d: list[tuple[float, float, float]] = []
    while explorer.More():
        vert = explorer.CurrentVertex()
        pnt = BRep_Tool.Pnt_s(vert)
        verts_3d.append((pnt.X(), pnt.Y(), pnt.Z()))
        explorer.Next()
    return verts_3d


def _project_panel(
    solid: cq.Solid,
    outer_normal: tuple[float, float, float],
    name: str,
    frame: Panel2D | None = None,
) -> Panel2D | None:
    wp = cq.Workplane().add(solid)
    faces = wp.faces().vals()

    best_face = None
    best_area = 0.0
    for f in faces:
        try:
            n = f.normalAt()
        except Exception:
            continue
        dot = n.x * outer_normal[0] + n.y * outer_normal[1] + n.z * outer_normal[2]
        if dot > 0.99:
            a = f.Area()
            if a > best_area:
                best_area = a
                best_face = f
    if best_face is None:
        return None

    outer_wire = best_face.outerWire()
    outer_verts_3d = _wire_vertices_3d(outer_wire)
    if not outer_verts_3d:
        return None

    hole_wires = list(best_face.innerWires())
    hole_verts_3d = [_wire_vertices_3d(w) for w in hole_wires]

    if frame is not None:
        # Project into an existing local frame so overlays align exactly.
        u = frame.u_axis
        v = frame.v_axis
        min_x = frame.offset_x
        min_y = frame.offset_y
        pts_2d = [(_vec_dot(pt, u) - min_x, _vec_dot(pt, v) - min_y) for pt in outer_verts_3d]
        holes_2d = [
            [(_vec_dot(pt, u) - min_x, _vec_dot(pt, v) - min_y) for pt in verts]
            for verts in hole_verts_3d
            if len(verts) >= 3
        ]
        return Panel2D(
            name=name, outline=pts_2d, holes=holes_2d,
            u_axis=u, v_axis=v,
            offset_x=min_x, offset_y=min_y,
        )

    u, v = _build_local_axes(outer_normal)
    raw_outline = [(_vec_dot(pt, u), _vec_dot(pt, v)) for pt in outer_verts_3d]
    raw_holes = [
        [(_vec_dot(pt, u), _vec_dot(pt, v)) for pt in verts]
        for verts in hole_verts_3d
        if len(verts) >= 3
    ]

    rot = _min_bbox_angle(raw_outline)
    outline_rot = _rotate_pts(raw_outline, -rot)
    holes_rot = [_rotate_pts(h, -rot) for h in raw_holes]

    min_x = min(p[0] for p in outline_rot)
    min_y = min(p[1] for p in outline_rot)
    pts_2d = [(p[0] - min_x, p[1] - min_y) for p in outline_rot]
    holes_2d = [
        [(p[0] - min_x, p[1] - min_y) for p in hole]
        for hole in holes_rot
        if len(hole) >= 3
    ]

    # Rotated 3D axes:
    #   final_x = dot(P, u)*cos(-rot) - dot(P, v)*sin(-rot) - min_x
    #   final_y = dot(P, u)*sin(-rot) + dot(P, v)*cos(-rot) - min_y
    cos_r = math.cos(-rot)
    sin_r = math.sin(-rot)
    u_rot = tuple(cos_r * u[k] - sin_r * v[k] for k in range(3))
    v_rot = tuple(sin_r * u[k] + cos_r * v[k] for k in range(3))

    return Panel2D(
        name=name, outline=pts_2d, holes=holes_2d,
        u_axis=u_rot, v_axis=v_rot,
        offset_x=min_x, offset_y=min_y,
    )


# ---------------------------------------------------------------------------
# Unfolded layout via BFS
# ---------------------------------------------------------------------------

def _build_adjacency(
    shared_edges: list[SharedEdge],
) -> dict[str, list[tuple[str, SharedEdge]]]:
    adj: dict[str, list[tuple[str, SharedEdge]]] = {}
    for se in shared_edges:
        adj.setdefault(se.panel_a, []).append((se.panel_b, se))
        adj.setdefault(se.panel_b, []).append((se.panel_a, se))
    return adj


def _outward_normal_2d(
    outline: list[tuple[float, float]],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """Unit normal of segment a->b pointing away from polygon interior."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (0.0, 1.0)
    n1 = (-dy / length, dx / length)
    n2 = (dy / length, -dx / length)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    eps = max(0.25, min(2.0, length * 0.05))

    p1 = (mid[0] + n1[0] * eps, mid[1] + n1[1] * eps)
    p2 = (mid[0] + n2[0] * eps, mid[1] + n2[1] * eps)
    inside1 = _point_in_polygon(p1, outline)
    inside2 = _point_in_polygon(p2, outline)

    if inside1 and not inside2:
        return n2
    if inside2 and not inside1:
        return n1

    # Fallback for ambiguous cases.
    cx = sum(p[0] for p in outline) / len(outline)
    cy = sum(p[1] for p in outline) / len(outline)
    to_c = (cx - mid[0], cy - mid[1])
    if n1[0] * to_c[0] + n1[1] * to_c[1] < 0:
        return n1
    return n2


def _compute_unfolded_layout(
    model: BinModel,
    panel_map: dict[str, Panel2D],
    gap: float = 4.0,
) -> list[tuple[str, list[tuple[float, float]], list[list[tuple[float, float]]], Affine2D]]:
    """BFS from the bottom panel, placing neighbours next to shared edges.

    Each neighbour is reflected across the shared edge and shifted outward
    by *gap* mm, as if the 3D box were unfolded flat.
    """
    adj = _build_adjacency(model.shared_edges)
    root = "bottom" if "bottom" in panel_map else next(iter(panel_map))

    # For each placed panel: final SVG outline + Affine2D that maps local -> SVG
    placed_outline: dict[str, list[tuple[float, float]]] = {}
    placed_holes: dict[str, list[list[tuple[float, float]]]] = {}
    placed_xform: dict[str, Affine2D] = {}

    # Root at the origin
    placed_outline[root] = list(panel_map[root].outline)
    placed_holes[root] = [list(h) for h in panel_map[root].holes]
    placed_xform[root] = Affine2D.identity()

    queue: deque[str] = deque([root])
    visited: set[str] = {root}

    while queue:
        current = queue.popleft()
        if current not in adj:
            continue

        cur_outline = placed_outline[current]
        cur_xform = placed_xform[current]
        cur_p2d = panel_map[current]

        for neighbor, se in adj[current]:
            if neighbor in visited or neighbor not in panel_map:
                continue

            nbr_p2d = panel_map[neighbor]

            # Project shared edge into both panels' local 2D
            se_cur_a = cur_p2d.project_3d(se.start_3d)
            se_cur_b = cur_p2d.project_3d(se.end_3d)
            se_nbr_a = nbr_p2d.project_3d(se.start_3d)
            se_nbr_b = nbr_p2d.project_3d(se.end_3d)

            # Transform current panel's SE to SVG coords
            se_svg_a = cur_xform.apply(*se_cur_a)
            se_svg_b = cur_xform.apply(*se_cur_b)
            se_svg_mid = ((se_svg_a[0] + se_svg_b[0]) / 2,
                          (se_svg_a[1] + se_svg_b[1]) / 2)

            # SE direction angles
            se_svg_dx = se_svg_b[0] - se_svg_a[0]
            se_svg_dy = se_svg_b[1] - se_svg_a[1]
            if math.hypot(se_svg_dx, se_svg_dy) < 1e-6:
                continue
            angle_svg = math.atan2(se_svg_dy, se_svg_dx)

            se_nbr_dx = se_nbr_b[0] - se_nbr_a[0]
            se_nbr_dy = se_nbr_b[1] - se_nbr_a[1]
            if math.hypot(se_nbr_dx, se_nbr_dy) < 1e-6:
                continue
            angle_nbr = math.atan2(se_nbr_dy, se_nbr_dx)

            out_n = _outward_normal_2d(cur_outline, se_svg_a, se_svg_b)

            # --- Build the neighbour's transform ---
            # Goal: rotate + reflect the neighbour so its SE aligns with the
            # SVG SE, and its body extends in the outward direction.
            #
            # 1. Rotate to align SE directions (make them parallel).
            # 2. Reflect across the SE direction so the body flips outward.
            # 3. Translate so SE midpoints are gap apart along outward normal.
            #
            # The composed transform (applied right-to-left):
            #   T_translate * T_reflect * T_rotate
            # maps neighbour-local -> SVG.

            align_rot = angle_svg - angle_nbr
            T_rot = Affine2D.from_rotation(align_rot)
            T_ref = Affine2D.from_reflection(angle_svg)
            T_rot_ref = T_ref.compose(T_rot)

            # Where does the SE midpoint end up after rot+ref?
            se_nbr_mid = ((se_nbr_a[0] + se_nbr_b[0]) / 2,
                          (se_nbr_a[1] + se_nbr_b[1]) / 2)
            xf_mid = T_rot_ref.apply(*se_nbr_mid)

            target = (se_svg_mid[0] + out_n[0] * gap,
                      se_svg_mid[1] + out_n[1] * gap)
            T_trans = Affine2D.from_translation(target[0] - xf_mid[0],
                                                target[1] - xf_mid[1])
            nbr_xform = T_trans.compose(T_rot_ref)

            final_outline = nbr_xform.apply_pts(nbr_p2d.outline)
            final_holes = [nbr_xform.apply_pts(h) for h in nbr_p2d.holes]

            # Verify neighbour body extends outward
            nbr_cx = sum(p[0] for p in final_outline) / len(final_outline)
            nbr_cy = sum(p[1] for p in final_outline) / len(final_outline)
            to_nbr = (nbr_cx - se_svg_mid[0], nbr_cy - se_svg_mid[1])
            body_outward = to_nbr[0] * out_n[0] + to_nbr[1] * out_n[1]

            if body_outward < 0:
                # Body ended up on wrong side -- use rotation only (no reflection)
                xf_mid2 = T_rot.apply(*se_nbr_mid)
                T_trans2 = Affine2D.from_translation(target[0] - xf_mid2[0],
                                                     target[1] - xf_mid2[1])
                nbr_xform = T_trans2.compose(T_rot)
                final_outline = nbr_xform.apply_pts(nbr_p2d.outline)
                final_holes = [nbr_xform.apply_pts(h) for h in nbr_p2d.holes]

            # If this panel still overlaps existing placed panels, push it farther
            # along the outward normal until clear.
            others = list(placed_outline.values())
            final_outline, pushed = _push_out_until_clear(
                final_outline,
                out_n,
                others,
                step=max(gap, 2.0),
                min_clearance=max(0.0, gap),
            )
            if pushed > 0:
                T_push = Affine2D.from_translation(out_n[0] * pushed, out_n[1] * pushed)
                nbr_xform = T_push.compose(nbr_xform)
                final_holes = [
                    _translate_pts(h, out_n[0] * pushed, out_n[1] * pushed)
                    for h in final_holes
                ]

            placed_outline[neighbor] = final_outline
            placed_holes[neighbor] = final_holes
            placed_xform[neighbor] = nbr_xform
            visited.add(neighbor)
            queue.append(neighbor)

    # Place unreachable panels below the rest
    for name in panel_map:
        if name not in placed_outline:
            p = panel_map[name]
            all_y = [pt[1] for pts in placed_outline.values() for pt in pts]
            y_off = max(all_y) + 20.0 if all_y else 0.0
            placed_outline[name] = _translate_pts(p.outline, 0, y_off)
            placed_holes[name] = [_translate_pts(h, 0, y_off) for h in p.holes]
            placed_xform[name] = Affine2D.from_translation(0, y_off)

    # Shift so all coordinates are positive
    all_x = [pt[0] for pts in placed_outline.values() for pt in pts]
    all_y = [pt[1] for pts in placed_outline.values() for pt in pts]
    if all_x and all_y:
        sx = -min(all_x) + 5.0
        sy = -min(all_y) + 5.0
        for name in placed_outline:
            placed_outline[name] = _translate_pts(placed_outline[name], sx, sy)
            placed_holes[name] = [_translate_pts(h, sx, sy) for h in placed_holes.get(name, [])]
            placed_xform[name] = Affine2D.from_translation(sx, sy).compose(placed_xform[name])

    return [
        (name, placed_outline[name], placed_holes.get(name, []), placed_xform[name])
        for name in placed_outline
    ]


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def export_svg(
    model: BinModel,
    output_path: str,
    spacing: float = 5.0,
    reference_model: BinModel | None = None,
) -> None:
    """Export all panels as flat 2D outlines into an SVG file.

    Uses an unfolded layout with 4 mm gaps between adjacent panels.
    """
    if svgwrite is None:
        raise ImportError("svgwrite is required for SVG export (pip install svgwrite)")

    panel_map: dict[str, Panel2D] = {}
    for name, panel in model.panels.items():
        p2d = _project_panel(panel.solid, panel.outer_normal, name)
        if p2d is not None:
            panel_map[name] = p2d

    if not panel_map:
        raise ValueError("No outlines could be projected")

    placed = _compute_unfolded_layout(model, panel_map, gap=4.0)
    if not placed:
        raise ValueError("No panels could be placed")

    all_x = [p[0] for _, pts, _, _ in placed for p in pts]
    all_y = [p[1] for _, pts, _, _ in placed for p in pts]
    total_w = max(all_x) + spacing
    total_h = max(all_y) + spacing

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{total_w}mm", f"{total_h}mm"),
        viewBox=f"0 0 {total_w} {total_h}",
    )

    # Explicit white background improves visibility in dark-theme viewers.
    dwg.add(dwg.rect(
        insert=(0, 0),
        size=(total_w, total_h),
        fill="#FFFFFF",
        stroke="none",
    ))

    for name, pts, holes, xform in placed:
        if not pts:
            continue
        d_parts = [f"M {pts[0][0]:.4f},{pts[0][1]:.4f}"]
        for p in pts[1:]:
            d_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
        d_parts.append("Z")

        dwg.add(dwg.path(
            d=" ".join(d_parts),
            stroke="#000000",
            stroke_width="0.1",
            fill="none",
        ))

        if reference_model is not None and name in reference_model.panels and name in panel_map:
            ref_panel = reference_model.panels[name]
            ref_p2d = _project_panel(
                ref_panel.solid,
                ref_panel.outer_normal,
                name,
                frame=panel_map[name],
            )
            if ref_p2d is not None and ref_p2d.outline:
                ref_outline = xform.apply_pts(ref_p2d.outline)
                r_parts = [f"M {ref_outline[0][0]:.4f},{ref_outline[0][1]:.4f}"]
                for p in ref_outline[1:]:
                    r_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
                r_parts.append("Z")
                dwg.add(dwg.path(
                    d=" ".join(r_parts),
                    stroke="#1BAA5C",
                    stroke_width="0.12",
                    stroke_dasharray="2.0,1.6",
                    fill="none",
                    opacity="0.9",
                ))

        for hole in holes:
            if len(hole) < 3:
                continue
            h_parts = [f"M {hole[0][0]:.4f},{hole[0][1]:.4f}"]
            for p in hole[1:]:
                h_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
            h_parts.append("Z")
            dwg.add(dwg.path(
                d=" ".join(h_parts),
                stroke="#000000",
                stroke_width="0.1",
                fill="none",
            ))

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        dwg.add(dwg.text(
            name, insert=(cx, cy),
            text_anchor="middle", dominant_baseline="central",
            font_size="4", fill="#222222",
        ))

    dwg.save()
