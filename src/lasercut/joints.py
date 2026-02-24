"""Finger joint generation via 3D boolean cuts on panel solids.

Supports two joint types:
- **Finger joints** for edge-joined panels (shared edge runs along both panels' boundaries)
- **Through-slots** for inset panels (shared edge cuts across one panel's face interior)
- **Living-hinge slit cuts** for shallow non-bottom seams (default: angle < 45 deg)
"""

from __future__ import annotations

import math

import cadquery as cq
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from lasercut.panels import (
    BinModel,
    Panel,
    SharedEdge,
    _extract_outer_wire_edges,
    _thicken_face_inward,
    _edge_overlap_length,
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


def _clone_panels(panels: dict[str, Panel]) -> dict[str, Panel]:
    """Clone panel metadata while keeping underlying CQ shapes."""
    cloned: dict[str, Panel] = {}
    for name, p in panels.items():
        cloned[name] = Panel(
            name=p.name,
            solid=p.solid,
            outer_normal=p.outer_normal,
            width=p.width,
            height=p.height,
            outer_face=p.outer_face,
            outer_edges=list(p.outer_edges),
        )
    return cloned


def _edge_endpoints(edge: cq.Edge) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return edge endpoints as tuples."""
    verts = edge.Vertices()
    if len(verts) >= 2:
        if hasattr(verts[0], "toVector") and hasattr(verts[1], "toVector"):
            p0 = verts[0].toVector()
            p1 = verts[1].toVector()
        elif hasattr(verts[0], "toTuple") and hasattr(verts[1], "toTuple"):
            t0 = verts[0].toTuple()
            t1 = verts[1].toTuple()
            return ((t0[0], t0[1], t0[2]), (t1[0], t1[1], t1[2]))
        else:
            c0 = verts[0].Center()
            c1 = verts[1].Center()
            return ((c0.x, c0.y, c0.z), (c1.x, c1.y, c1.z))
        return ((p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z))

    p0 = edge.positionAt(0)
    p1 = edge.positionAt(1)
    return ((p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z))


def _match_source_edges_for_shared_edges(
    source_solid: cq.Shape,
    shared_edges: list[SharedEdge],
    line_tolerance: float = 1.5,
) -> list[tuple[cq.Edge, SharedEdge]]:
    """Match extracted shared-edge segments to line edges in the source solid."""
    source_edges = [e for e in source_solid.Edges() if e.geomType() == "LINE"]
    matched: list[tuple[cq.Edge, SharedEdge]] = []
    used: set[int] = set()

    for se in shared_edges:
        se_vec = _vec_sub(se.end_3d, se.start_3d)
        se_len = _vec_len(se_vec)
        if se_len < 1e-6:
            continue
        se_dir = _normalize(se_vec)

        best_idx: int | None = None
        best_overlap = 0.0
        for idx, edge in enumerate(source_edges):
            if idx in used:
                continue
            p0, p1 = _edge_endpoints(edge)
            edge_vec = _vec_sub(p1, p0)
            edge_len = _vec_len(edge_vec)
            if edge_len < 1e-6:
                continue
            edge_dir = _normalize(edge_vec)
            if abs(_vec_dot(se_dir, edge_dir)) < 0.97:
                continue

            d0 = _point_to_line_dist(p0, se.start_3d, se.end_3d)
            d1 = _point_to_line_dist(p1, se.start_3d, se.end_3d)
            if d0 > line_tolerance and d1 > line_tolerance:
                continue

            overlap = _edge_overlap_length(se.start_3d, se.end_3d, p0, p1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        if best_idx is None:
            continue

        min_overlap = max(5.0, se.edge_length * 0.35)
        if best_overlap < min_overlap:
            continue

        matched.append((source_edges[best_idx], se))
        used.add(best_idx)

    return matched


def _make_finger_joint_faces_safe(
    shape: cq.Shape,
    finger_joint_edges: list[cq.Edge],
    seam_by_edge: dict[cq.Edge, SharedEdge] | None,
    material_thickness: float,
    target_finger_width: float,
    kerf_width: float = 0.0,
) -> list[cq.Face]:
    """cq_warehouse-style finger-joint face generation for selected seam edges.

    Mirrors cq_warehouse logic but supports partial edge selections safely.
    """
    working_faces = shape.Faces()
    working_face_areas = [f.Area() for f in working_faces]

    edge_adjacency: dict[cq.Edge, list[int]] = {}
    edge_vertex_adjacency: dict[cq.Vertex, set[int]] = {}
    filtered_edges: list[cq.Edge] = []

    for common_edge in finger_joint_edges:
        if common_edge.Length() < max(0.5, target_finger_width * 0.5):
            continue
        adjacent_face_indices = [
            i for i, face in enumerate(working_faces) if common_edge in face.Edges()
        ]
        if len(adjacent_face_indices) != 2:
            continue

        filtered_edges.append(common_edge)
        edge_adjacency[common_edge] = adjacent_face_indices
        for v in common_edge.Vertices():
            if v in edge_vertex_adjacency:
                edge_vertex_adjacency[v].update(adjacent_face_indices)
            else:
                edge_vertex_adjacency[v] = set(adjacent_face_indices)

    if not edge_adjacency:
        raise RuntimeError("No valid seam edges for cq_warehouse face jointing")

    finger_depths: dict[cq.Edge, float] = {}
    external_corners: dict[cq.Edge, bool] = {}
    for common_edge, adjacent_face_indices in edge_adjacency.items():
        face_centers = [working_faces[i].Center() for i in adjacent_face_indices]
        face_normals = [
            working_faces[i].normalAt(working_faces[i].Center())
            for i in adjacent_face_indices
        ]
        ref_plane = cq.Plane(origin=face_centers[0], normal=face_normals[0])
        localized_opposite_center = ref_plane.toLocalCoords(face_centers[1])
        external_corners[common_edge] = localized_opposite_center.z < 0

        corner_angle = abs(
            face_normals[0].getSignedAngle(face_normals[1], common_edge.tangentAt(0))
        )
        finger_depths[common_edge] = material_thickness * max(
            math.sin(corner_angle),
            (
                math.sin(corner_angle)
                + (math.cos(corner_angle) - 1) * math.tan(math.pi / 2 - corner_angle)
            ),
        )

    vertices_with_internal_edge: dict[cq.Vertex, bool] = {}
    for e in filtered_edges:
        for v in e.Vertices():
            if v in vertices_with_internal_edge:
                vertices_with_internal_edge[v] = (
                    vertices_with_internal_edge[v] or not external_corners[e]
                )
            else:
                vertices_with_internal_edge[v] = not external_corners[e]

    # Faces may include vertices not present in selected seam edges.
    open_internal_vertices: dict[cq.Vertex, set[int]] = {}
    for i, face in enumerate(working_faces):
        for v in face.Vertices():
            if vertices_with_internal_edge.get(v, False):
                if i not in edge_vertex_adjacency.get(v, set()):
                    if v in open_internal_vertices:
                        open_internal_vertices[v].add(i)
                    else:
                        open_internal_vertices[v] = {i}

    for common_edge, adjacent_face_indices in edge_adjacency.items():
        corner_face_counter: dict[cq.Vertex, set[int]] = {}
        primary_face_index = adjacent_face_indices[0]
        secondary_face_index = adjacent_face_indices[1]
        reverse_align = False
        if seam_by_edge is not None and common_edge in seam_by_edge:
            se = seam_by_edge[common_edge]
            # Keep specific seams in opposite phase to avoid corner remnants
            # where the back lip and side-wall seams converge.
            reverse_align = {se.panel_a, se.panel_b} in (
                {"bottom", "back_wall"},
                {"right_wall", "back_wall"},
                {"left_wall", "back_wall"},
            )

        if reverse_align:
            if working_face_areas[primary_face_index] < working_face_areas[secondary_face_index]:
                primary_face_index, secondary_face_index = secondary_face_index, primary_face_index
        else:
            if working_face_areas[primary_face_index] > working_face_areas[secondary_face_index]:
                primary_face_index, secondary_face_index = secondary_face_index, primary_face_index

        for i in [primary_face_index, secondary_face_index]:
            working_faces[i] = working_faces[i].makeFingerJoints(
                common_edge,
                finger_depths[common_edge],
                target_finger_width,
                corner_face_counter,
                open_internal_vertices,
                alignToBottom=i == primary_face_index,
                externalCorner=external_corners[common_edge],
                faceIndex=i,
            )

    tabbed_face_indices = set(i for face_list in edge_adjacency.values() for i in face_list)
    tabbed_faces = [working_faces[i] for i in tabbed_face_indices]

    if kerf_width != 0.0:
        tabbed_faces = [
            cq.Face.makeFromWires(
                f.outerWire().offset2D(kerf_width / 2)[0],
                [i.offset2D(-kerf_width / 2)[0] for i in f.innerWires()],
            )
            for f in tabbed_faces
        ]

    return tabbed_faces


def _apply_finger_joints_cqwarehouse(
    model: BinModel,
    finger_width: float,
    kerf: float = 0.0,
    living_hinge_angle_threshold_deg: float = 45.0,
) -> BinModel:
    """Use cq_warehouse topology-aware finger joints when source solid is available."""
    if model.source_solid is None:
        raise ValueError("No source_solid available for cq_warehouse processing")

    try:
        import cq_warehouse.extensions  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("cq_warehouse is not installed") from exc

    # Classify seams first so inset seams can be treated as through-slots and
    # shallow non-bottom seams can be switched to living hinges.
    finger_seams: list[SharedEdge] = []
    through_slot_seams: list[tuple[SharedEdge, str]] = []
    hinge_seams: list[SharedEdge] = []
    for se in model.shared_edges:
        jt, slot_panel = _classify_joint_type(se, model.panels)
        if jt == "through_slot" and slot_panel is not None:
            through_slot_seams.append((se, slot_panel))
        elif jt == "finger" and _should_use_living_hinge(
            se, model.panels, living_hinge_angle_threshold_deg
        ):
            hinge_seams.append(se)
        else:
            finger_seams.append(se)

    panels = _clone_panels(model.panels)
    joint_edges: list[cq.Edge] = []
    matched_names: set[str] = set()

    if finger_seams:
        source_solid = _to_cuttable(model.source_solid)
        matched_edge_seams = _match_source_edges_for_shared_edges(source_solid, finger_seams)
        joint_edges = [edge for edge, _ in matched_edge_seams]
        seam_by_edge = {edge: se for edge, se in matched_edge_seams}
        if not joint_edges:
            raise RuntimeError("No source edges matched shared seams for cq_warehouse")

        jointed_faces = _make_finger_joint_faces_safe(
            source_solid,
            joint_edges,
            seam_by_edge=seam_by_edge,
            material_thickness=model.thickness,
            target_finger_width=finger_width,
            kerf_width=kerf,
        )
        if not jointed_faces:
            raise RuntimeError("cq_warehouse returned no jointed faces")

        panel_reference: list[tuple[str, Panel, tuple[float, float, float], float]] = []
        for name, panel in panels.items():
            if panel.outer_face is None:
                continue
            c = panel.outer_face.Center()
            panel_reference.append((name, panel, (c.x, c.y, c.z), panel.outer_face.Area()))

        # Build one-to-one mapping panel -> face.
        used_faces: set[int] = set()
        for name, panel, ref_center, ref_area in panel_reference:
            best_idx: int | None = None
            best_score = float("-inf")
            for idx, face in enumerate(jointed_faces):
                if idx in used_faces:
                    continue
                fc = face.Center()
                fn = face.normalAt(fc)
                fn_t = (fn.x, fn.y, fn.z)
                normal_align = abs(_vec_dot(fn_t, panel.outer_normal))
                if normal_align < 0.75:
                    continue

                face_center = (fc.x, fc.y, fc.z)
                center_dist = _pt_dist(face_center, ref_center)
                area_delta = abs(face.Area() - ref_area)
                score = normal_align * 1000.0 - center_dist * 2.0 - area_delta * 0.15
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                continue

            used_faces.add(best_idx)
            face = jointed_faces[best_idx]
            base_panel = panels[name]
            new_edges = _extract_outer_wire_edges(face)
            new_solid = _thicken_face_inward(face, base_panel.outer_normal, model.thickness)
            panels[name] = Panel(
                name=base_panel.name,
                solid=new_solid,
                outer_normal=base_panel.outer_normal,
                width=base_panel.width,
                height=base_panel.height,
                outer_face=face,
                outer_edges=new_edges,
            )
            matched_names.add(name)

        if not matched_names:
            raise RuntimeError("cq_warehouse faces could not be matched back to panels")

    corners = _find_corner_points(model.shared_edges)

    # Apply inset through-slots after face-based finger generation.
    if through_slot_seams:
        for se, slot_panel_name in through_slot_seams:
            print(f"  Through-slot: {se.panel_a}--{se.panel_b} (slot in {slot_panel_name})")
            _apply_through_slot(
                se,
                slot_panel_name,
                panels,
                corners,
                model.thickness,
                finger_width=finger_width,
                kerf=kerf,
            )

    if hinge_seams:
        for se in hinge_seams:
            angle = _seam_panel_angle_deg(se, panels)
            print(
                f"  Living-hinge seam: {se.panel_a}--{se.panel_b} "
                f"(angle={angle:.1f} deg < {living_hinge_angle_threshold_deg:.1f})"
            )

    trimmed_seams = _trim_side_wall_overhangs_against_back_wall(
        model.shared_edges,
        panels,
        corners,
        model.thickness,
        kerf=kerf,
    )

    print(
        f"  cq_warehouse: jointed {len(matched_names)} panels "
        f"from {len(joint_edges)} edges ({len(through_slot_seams)} through-slots, "
        f"{len(hinge_seams)} living-hinge seams, "
        f"{trimmed_seams} side-trimmed seams)"
    )
    return BinModel(
        panels=panels,
        shared_edges=model.shared_edges,
        thickness=model.thickness,
        source_solid=model.source_solid,
        living_hinge_seams=hinge_seams,
    )


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
# Shallow-angle living hinge selection
# ---------------------------------------------------------------------------

def _is_bottom_panel_name(name: str) -> bool:
    return name == "bottom" or name.startswith("bottom_")


def _seam_panel_angle_deg(se: SharedEdge, panels: dict[str, Panel]) -> float:
    """Return acute angle between seam panels in degrees (0..90)."""
    pa = panels[se.panel_a]
    pb = panels[se.panel_b]
    na = _normalize(pa.outer_normal)
    nb = _normalize(pb.outer_normal)
    d = max(-1.0, min(1.0, abs(_vec_dot(na, nb))))
    return math.degrees(math.acos(d))


def _should_use_living_hinge(
    se: SharedEdge,
    panels: dict[str, Panel],
    angle_threshold_deg: float,
) -> bool:
    """True when a seam should be handled as living hinge instead of finger tabs."""
    if angle_threshold_deg <= 0:
        return False
    if _is_bottom_panel_name(se.panel_a) or _is_bottom_panel_name(se.panel_b):
        return False
    if se.panel_a not in panels or se.panel_b not in panels:
        return False
    return _seam_panel_angle_deg(se, panels) < angle_threshold_deg


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


def _merge_intervals(intervals: list[tuple[float, float]], tol: float = 0.05) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _complement_intervals(
    lo: float,
    hi: float,
    keep: list[tuple[float, float]],
    tol: float = 0.05,
) -> list[tuple[float, float]]:
    """Return intervals in [lo, hi] not covered by *keep*."""
    if hi - lo <= tol:
        return []

    clipped = [
        (max(lo, a), min(hi, b))
        for a, b in keep
        if min(hi, b) - max(lo, a) > tol
    ]
    clipped = _merge_intervals(clipped, tol=tol)

    out: list[tuple[float, float]] = []
    cur = lo
    for a, b in clipped:
        if a - cur > tol:
            out.append((cur, a))
        cur = max(cur, b)
    if hi - cur > tol:
        out.append((cur, hi))
    return out


def _inset_slot_intervals_from_lip(
    se: SharedEdge,
    slot_panel: Panel,
    slot_start: tuple[float, float, float],
    slot_end: tuple[float, float, float],
    edge_dir: tuple[float, float, float],
    slot_in_plane: tuple[float, float, float],
    thickness: float,
    finger_width: float,
    start_keepout: float,
    end_keepout: float,
    segment_end_trim: float = 3.5,
) -> list[tuple[float, float]]:
    """Find slot intervals where the slot-panel lip is full-depth (not notched).

    For inset seams with a notched lip, we only cut through-slot segments where
    the outer lip is present. This avoids removing bridges at lip-notch locations.
    """
    d_len = _vec_len(_vec_sub(slot_end, slot_start))
    if d_len < 1e-6:
        return []

    # Gather seam-parallel boundary segments near the seam on the lip/outside side.
    raw: list[tuple[float, float, float]] = []  # (u, lo, hi)
    search_band = max(thickness * 3.5, 12.0)
    for p0, p1 in slot_panel.outer_edges:
        pe = _vec_sub(p1, p0)
        pe_len = _vec_len(pe)
        if pe_len < 0.2:
            continue
        pe_dir = _normalize(pe)
        if abs(_vec_dot(pe_dir, edge_dir)) < 0.95:
            continue

        v0 = _vec_sub(p0, slot_start)
        v1 = _vec_sub(p1, slot_start)
        t0 = _vec_dot(v0, edge_dir)
        t1 = _vec_dot(v1, edge_dir)
        u0 = _vec_dot(v0, slot_in_plane)
        u1 = _vec_dot(v1, slot_in_plane)

        # Nearly constant offset from seam, i.e. seam-parallel lip segment.
        if abs(u0 - u1) > max(0.8, thickness * 0.4):
            continue
        u = (u0 + u1) / 2.0
        if u > 0.4:
            continue  # interior side; seam lip is on negative side
        if abs(u) > search_band:
            continue

        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        lo = max(0.0, lo)
        hi = min(d_len, hi)
        if hi - lo > 0.2:
            raw.append((u, lo, hi))

    if not raw:
        return []

    # Full-depth lip segments are the farthest outward (most negative u).
    u_outer = min(u for u, _, _ in raw)
    u_cutoff = u_outer + max(0.8, thickness * 0.5)
    selected = [(lo, hi) for u, lo, hi in raw if u <= u_cutoff]
    selected = _merge_intervals(selected)

    # Apply corner keepouts and drop tiny fragments.
    usable_lo = start_keepout
    usable_hi = d_len - end_keepout
    min_seg = max(12.0, finger_width * 0.9)
    clipped: list[tuple[float, float]] = []
    for lo, hi in selected:
        lo = max(lo, usable_lo)
        hi = min(hi, usable_hi)
        if hi - lo >= min_seg:
            clipped.append((lo, hi))

    # Trim each segment's ends so tabs/slots stay clear of nearby notch valleys.
    if segment_end_trim > 0:
        trimmed: list[tuple[float, float]] = []
        min_after_trim = max(8.0, finger_width * 0.45)
        for lo, hi in clipped:
            tlo = lo + segment_end_trim
            thi = hi - segment_end_trim
            if thi - tlo >= min_after_trim:
                trimmed.append((tlo, thi))
        if trimmed:
            return trimmed

    return clipped


def _is_side_wall_name(name: str) -> bool:
    return (
        name == "right_wall"
        or name == "left_wall"
        or name.startswith("right_wall_")
        or name.startswith("left_wall_")
    )


def _trim_side_wall_overhangs_against_back_wall(
    shared_edges: list[SharedEdge],
    panels: dict[str, Panel],
    corners: list[tuple[tuple[float, float, float], set[str]]],
    thickness: float,
    kerf: float = 0.0,
    end_trim: float = 3.5,
) -> int:
    """Trim side-wall seam-end overhangs on back-wall joints.

    This removes tiny corner remnants created where the inset back lip extends
    past the effective back-wall engagement zone.
    """
    if end_trim <= 0:
        return 0

    trimmed_count = 0
    cut_width = max(0.2, thickness - kerf)

    for se in shared_edges:
        side_name: str | None = None
        if _is_side_wall_name(se.panel_a) and se.panel_b == "back_wall":
            side_name = se.panel_a
        elif _is_side_wall_name(se.panel_b) and se.panel_a == "back_wall":
            side_name = se.panel_b

        if side_name is None or side_name not in panels:
            continue

        side_panel = panels[side_name]
        seam_start, seam_end = _project_edge_to_panel(se, side_panel)
        seam_vec = _vec_sub(seam_end, seam_start)
        seam_len = _vec_len(seam_vec)
        if seam_len < 2.0:
            continue

        trim_len = min(end_trim, seam_len * 0.3)
        if trim_len < 0.6:
            continue

        start_keepout, end_keepout = _corner_keepout_for_edge(se, corners, thickness)
        offsets: list[float] = []
        if start_keepout > 0:
            offsets.append(0.0)
        if end_keepout > 0:
            offsets.append(seam_len - trim_len)
        if not offsets:
            continue

        edge_dir = _normalize(seam_vec)
        in_plane = _edge_inward_direction(side_panel, seam_start, seam_end)
        into_side = _normalize((
            -side_panel.outer_normal[0],
            -side_panel.outer_normal[1],
            -side_panel.outer_normal[2],
        ))

        solid = _to_cuttable(side_panel.solid)
        for offset in offsets:
            origin = _add(seam_start, _scale(edge_dir, offset))
            trim_box = _make_oriented_box(
                origin=origin,
                x_dir=edge_dir,
                y_dir=in_plane,
                z_dir=into_side,
                dx=trim_len,
                dy=cut_width,
                dz=thickness * 2,
            )
            solid = _to_cuttable(solid.cut(trim_box))

        side_panel.solid = solid
        trimmed_count += 1

    return trimmed_count


def _apply_living_hinge_on_seam(
    se: SharedEdge,
    panels: dict[str, Panel],
    corners: list[tuple[tuple[float, float, float], set[str]]],
    thickness: float,
    kerf: float = 0.0,
) -> int:
    """Cut a slit pattern near the seam on both connected panels.

    This intentionally replaces interlocking fingers on shallow-angle seams
    with a flexible relief zone.
    """
    slit_count = 0
    slit_width = max(0.8, min(1.4, thickness * 0.35))
    bridge_width = max(1.8, thickness * 0.65)
    slit_pitch = slit_width + bridge_width
    slit_depth = max(6.0, thickness * 2.4)
    seam_inset = max(0.6, thickness * 0.2)
    end_margin = max(2.0, thickness * 0.75)

    # Positive kerf means tighter fit generally. For hinge slits keep an almost
    # neutral compensation to avoid over-weakening at high kerf values.
    effective_slit_width = max(0.4, slit_width - max(0.0, kerf) * 0.25)

    for panel_name in (se.panel_a, se.panel_b):
        panel = panels.get(panel_name)
        if panel is None:
            continue

        seam_start, seam_end = _project_edge_to_panel(se, panel)
        seam_vec = _vec_sub(seam_end, seam_start)
        seam_len = _vec_len(seam_vec)
        if seam_len < 8.0:
            continue

        edge_dir = _normalize(seam_vec)
        in_plane = _edge_inward_direction(panel, seam_start, seam_end)
        into_panel = _normalize((
            -panel.outer_normal[0],
            -panel.outer_normal[1],
            -panel.outer_normal[2],
        ))

        start_keepout, end_keepout = _corner_keepout_for_edge(se, corners, thickness)
        usable_start = start_keepout + end_margin
        usable_end = seam_len - end_keepout - end_margin
        usable_len = usable_end - usable_start
        if usable_len < max(10.0, slit_pitch * 1.2):
            continue

        n_slits = max(1, int((usable_len + bridge_width) / slit_pitch))
        pattern_len = n_slits * effective_slit_width + (n_slits - 1) * bridge_width
        base_offset = usable_start + max(0.0, (usable_len - pattern_len) / 2.0)

        solid = _to_cuttable(panel.solid)
        for i in range(n_slits):
            offset = base_offset + i * slit_pitch
            origin = _add(seam_start, _scale(edge_dir, offset))
            origin = _add(origin, _scale(in_plane, seam_inset))
            slit_box = _make_oriented_box(
                origin=origin,
                x_dir=edge_dir,
                y_dir=in_plane,
                z_dir=into_panel,
                dx=effective_slit_width,
                dy=slit_depth,
                dz=thickness * 2.0,
            )
            solid = _to_cuttable(solid.cut(slit_box))
            slit_count += 1

        panel.solid = solid

    return slit_count


# ---------------------------------------------------------------------------
# Main joint application
# ---------------------------------------------------------------------------

def _apply_through_slot(
    se: SharedEdge,
    slot_panel_name: str,
    panels: dict[str, Panel],
    corners: list[tuple[tuple[float, float, float], set[str]]],
    thickness: float,
    finger_width: float = 20.0,
    kerf: float = 0.0,
) -> None:
    """Cut a through-slot in one panel and matching tabs on the other.

    The *slot_panel_name* panel gets one or more through-slot segments.
    The other panel is recessed between those segments so only matching tabs remain.

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
        wall_panel = panels[se.panel_b]
    else:
        slot_panel = panels[se.panel_b]
        wall_panel = panels[se.panel_a]

    # Edge direction in slot-panel plane
    slot_start, slot_end = _project_edge_to_panel(se, slot_panel)
    d = _vec_sub(slot_end, slot_start)
    d_len = _vec_len(d)
    if d_len < 1e-6:
        return
    edge_dir = _normalize(d)

    # Keepouts at corners
    start_keepout, end_keepout = _corner_keepout_for_edge(se, corners, thickness)

    usable_start = start_keepout
    usable_end = d_len - end_keepout
    if usable_end - usable_start <= 0:
        return

    # Direction through the slot panel thickness (perpendicular to slot panel face)
    into_slot = _normalize((
        -slot_panel.outer_normal[0],
        -slot_panel.outer_normal[1],
        -slot_panel.outer_normal[2],
    ))

    # In-plane direction from shared edge into slot panel interior.
    slot_in_plane = _edge_inward_direction(slot_panel, slot_start, slot_end)

    slot_intervals = _inset_slot_intervals_from_lip(
        se=se,
        slot_panel=slot_panel,
        slot_start=slot_start,
        slot_end=slot_end,
        edge_dir=edge_dir,
        slot_in_plane=slot_in_plane,
        thickness=thickness,
        finger_width=finger_width,
        start_keepout=start_keepout,
        end_keepout=end_keepout,
    )

    if not slot_intervals:
        # Fallback to one continuous slot if lip-based segmentation is unavailable.
        slot_intervals = [(usable_start, usable_end)]

    # Positive kerf compensation tightens fit: shrink slot/recess cut width.
    cut_width = max(0.2, thickness - kerf)

    # Cut through-slot segments in the slot panel.
    solid_slot = _to_cuttable(slot_panel.solid)
    for lo, hi in slot_intervals:
        slot_length = hi - lo
        if slot_length <= 0:
            continue
        slot_origin = _add(slot_start, _scale(edge_dir, lo))
        slot_box = _make_oriented_box(
            origin=slot_origin,
            x_dir=edge_dir,
            y_dir=slot_in_plane,
            z_dir=into_slot,
            dx=slot_length,
            dy=cut_width,
            dz=thickness * 2,  # overshoot to ensure full penetration
        )
        solid_slot = _to_cuttable(solid_slot.cut(slot_box))

    slot_panel.solid = solid_slot

    # Recess non-slot regions on the wall panel so only matching tabs remain.
    wall_start, wall_end = _project_edge_to_panel(se, wall_panel)
    wall_vec = _vec_sub(wall_end, wall_start)
    wall_len = _vec_len(wall_vec)
    if wall_len < 1e-6:
        return
    wall_dir = _normalize(wall_vec)

    # Keep wall and slot param directions aligned so interval reuse is valid.
    if _vec_dot(wall_dir, edge_dir) < 0:
        wall_start, wall_end = wall_end, wall_start
        wall_dir = (-wall_dir[0], -wall_dir[1], -wall_dir[2])

    # Keep only explicit tab intervals on the wall; recess everything else.
    # This avoids tiny stray tabs at seam ends caused by corner keepouts.
    wall_usable_start = 0.0
    wall_usable_end = wall_len
    if wall_usable_end - wall_usable_start <= 0:
        return

    recess_intervals = _complement_intervals(
        wall_usable_start,
        wall_usable_end,
        slot_intervals,
    )
    if not recess_intervals:
        return

    wall_in_plane = _edge_inward_direction(wall_panel, wall_start, wall_end)
    into_wall = _normalize((
        -wall_panel.outer_normal[0],
        -wall_panel.outer_normal[1],
        -wall_panel.outer_normal[2],
    ))

    solid_wall = _to_cuttable(wall_panel.solid)
    boundary_overcut = 0.2
    for lo, hi in recess_intervals:
        cut_len = hi - lo
        if cut_len <= 0:
            continue
        cut_origin = _add(wall_start, _scale(wall_dir, lo))
        # Nudge outward across the seam boundary to avoid zero-width skins that
        # can otherwise show up as stray "inner hole" loops in 2D projection.
        cut_origin = _add(cut_origin, _scale(wall_in_plane, -boundary_overcut))
        recess_box = _make_oriented_box(
            origin=cut_origin,
            x_dir=wall_dir,
            y_dir=wall_in_plane,
            z_dir=into_wall,
            dx=cut_len,
            dy=cut_width + 2.0 * boundary_overcut,
            dz=thickness * 2,
        )
        solid_wall = _to_cuttable(solid_wall.cut(recess_box))

    wall_panel.solid = solid_wall


def apply_finger_joints(
    model: BinModel,
    finger_width: float = 20.0,
    kerf: float = 0.0,
    living_hinge_angle_threshold_deg: float = 45.0,
) -> BinModel:
    """Apply finger joints at all shared edges by boolean-cutting panel solids.

    For edge-joined panels (normal case):
    - Panel A gets tabs (fingers protruding), Panel B gets slots.
    - At even-indexed positions (0, 2, 4, ...): Panel A keeps material, Panel B gets cut
    - At odd-indexed positions (1, 3, 5, ...): Panel A gets cut, Panel B keeps material

    For inset panels (shared edge crosses one panel's face):
    - The crossed panel gets one or more through-slot segments
    - The inset wall panel is recessed between segments to form matching tabs

    For shallow non-bottom seams (angle between panel normals below threshold):
    - Finger joints are skipped
    - A living-hinge slit pattern is cut near the seam in both panels

    Returns a new BinModel with modified panel solids.

    Kerf sign convention:
    - positive kerf => tighter fit (smaller slots / larger tabs)
    - negative kerf => looser fit
    """
    # Prefer topology-aware jointing from cq_warehouse when a source solid exists.
    if model.source_solid is not None:
        try:
            return _apply_finger_joints_cqwarehouse(
                model,
                finger_width,
                kerf=kerf,
                living_hinge_angle_threshold_deg=living_hinge_angle_threshold_deg,
            )
        except Exception as exc:
            print(f"  cq_warehouse fallback to custom joints ({exc})")

    thickness = model.thickness
    panels = _clone_panels(model.panels)

    corners = _find_corner_points(model.shared_edges)
    hinge_seams: list[SharedEdge] = []

    for se in model.shared_edges:
        # Classify the joint type
        joint_type, slot_panel_name = _classify_joint_type(se, panels)

        if joint_type == "through_slot" and slot_panel_name is not None:
            print(f"  Through-slot: {se.panel_a}--{se.panel_b} (slot in {slot_panel_name})")
            _apply_through_slot(
                se,
                slot_panel_name,
                panels,
                corners,
                thickness,
                finger_width=finger_width,
                kerf=kerf,
            )
            continue

        if joint_type == "finger" and _should_use_living_hinge(
            se, panels, living_hinge_angle_threshold_deg
        ):
            angle = _seam_panel_angle_deg(se, panels)
            print(
                f"  Living-hinge: {se.panel_a}--{se.panel_b} "
                f"(angle={angle:.1f} deg < {living_hinge_angle_threshold_deg:.1f})"
            )
            hinge_seams.append(se)
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
        # Positive kerf shrinks slot cuts for tighter press-fit.
        slot_cut_width = max(0.2, thickness - kerf)

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
                    dy=slot_cut_width,
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
                    dy=slot_cut_width,
                    dz=thickness,
                )
                result = solid_a.cut(box)
                solid_a = _to_cuttable(result)

        pa.solid = solid_a
        pb.solid = solid_b

    _trim_side_wall_overhangs_against_back_wall(
        model.shared_edges,
        panels,
        corners,
        thickness,
        kerf=kerf,
    )

    return BinModel(
        panels=panels,
        shared_edges=model.shared_edges,
        thickness=thickness,
        source_solid=model.source_solid,
        living_hinge_seams=hinge_seams,
    )
