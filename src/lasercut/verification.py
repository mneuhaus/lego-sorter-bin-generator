"""Joint mesh verification helpers.

Provides a fast 2D seam-complement check and an optional interference proxy.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import svgwrite
from shapely.affinity import affine_transform
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, Point, box
from shapely.prepared import prep
from shapely.validation import make_valid

from .face_classifier import SharedEdge
from .finger_joints import (
    _find_bottom_edge_endpoints,
    _find_matching_edge_index,
    _outward_direction,
    _polygon_to_shapely,
)
from .projector import Projection2D, _project_point
from .step_loader import PlanarFace


@dataclass
class JointVerification:
    """Verification result for one joint edge pair."""

    joint_id: str
    joint_type: str
    face_a_id: int
    face_b_id: int
    samples: int
    reversed_b: bool
    shift_samples: int
    mismatch_ratio: float
    collision_ratio: float
    double_slot_ratio: float
    add_coverage_a: float
    add_coverage_b: float
    sub_coverage_a: float
    sub_coverage_b: float
    passed: bool
    reason: str
    debug_svg: str | None = None


@dataclass
class VerificationReport:
    """Aggregate report for all joints."""

    total_joints: int
    failed_joints: int
    run_interference: bool
    mismatch_tolerance: float
    interference_tolerance: float
    joints: list[JointVerification]

    @property
    def passed(self) -> bool:
        return self.failed_joints == 0


def _shape_from_projection(proj: Projection2D):
    return _polygon_to_shapely(proj.outer_polygon, proj.inner_polygons)


def _shape_from_modified(
    face_id: int,
    modified_polygons: dict[int, list[tuple[float, float]]],
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None,
):
    if slot_cutouts is None:
        slot_cutouts = {}
    return _polygon_to_shapely(modified_polygons.get(face_id, []), slot_cutouts.get(face_id, []))


def _sanitize_geom(geom):
    """Return a valid non-empty geometry when possible."""
    if geom.is_empty:
        return geom
    g = geom
    if not g.is_valid:
        try:
            g = make_valid(g)
        except Exception:
            pass
    if not g.is_valid:
        try:
            g = g.buffer(0)
        except Exception:
            pass
    return g


def _safe_difference(a, b):
    a2 = _sanitize_geom(a)
    b2 = _sanitize_geom(b)
    try:
        return a2.difference(b2)
    except GEOSException:
        try:
            return a2.buffer(0).difference(b2.buffer(0))
        except Exception:
            return a2


def _safe_intersection(a, b):
    a2 = _sanitize_geom(a)
    b2 = _sanitize_geom(b)
    try:
        return a2.intersection(b2)
    except GEOSException:
        try:
            return a2.buffer(0).intersection(b2.buffer(0))
        except Exception:
            return GeometryCollection()


def _ratio_true(mask: list[bool]) -> float:
    if not mask:
        return 0.0
    return sum(1 for v in mask if v) / len(mask)


def _xor_ratio(a: list[bool], b: list[bool]) -> float:
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def _and_ratio(a: list[bool], b: list[bool]) -> float:
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x and y) / len(a)


def _masked_xor_ratio(a: list[bool], b: list[bool], mask: list[bool]) -> float:
    n = 0
    bad = 0
    for x, y, m in zip(a, b, mask):
        if not m:
            continue
        n += 1
        if x != y:
            bad += 1
    if n == 0:
        return 0.0
    return bad / n


def _reverse_mask(mask: list[bool]) -> list[bool]:
    return list(reversed(mask))


def _shift_mask(mask: list[bool], shift: int) -> list[bool]:
    """Shift mask with zero-fill. Positive shift moves content to the right."""
    n = len(mask)
    if n == 0 or shift == 0:
        return list(mask)
    out = [False] * n
    if shift > 0:
        out[shift:] = mask[: n - shift]
    else:
        s = -shift
        out[: n - s] = mask[s:]
    return out


def _local_affine_for_edge(
    p1: tuple[float, float],
    p2: tuple[float, float],
    outward: tuple[float, float],
) -> tuple[list[float], float]:
    """Return Shapely affine matrix mapping global XY -> edge local frame.

    Local frame:
    - x axis along p1->p2
    - y axis along outward normal
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    edge_len = math.hypot(dx, dy)
    if edge_len < 1e-9:
        return [1, 0, 0, 1, -p1[0], -p1[1]], 0.0

    ux = dx / edge_len
    uy = dy / edge_len
    vx, vy = outward

    xoff = -(p1[0] * ux + p1[1] * uy)
    yoff = -(p1[0] * vx + p1[1] * vy)
    matrix = [ux, uy, vx, vy, xoff, yoff]
    return matrix, edge_len


def _edge_feature_masks(
    raw_shape,
    mod_shape,
    p1: tuple[float, float],
    p2: tuple[float, float],
    outward: tuple[float, float],
    thickness: float,
    samples: int,
) -> tuple[list[bool], list[bool]]:
    """Sample tab/slot presence along an edge.

    Returns:
        (tab_mask, slot_mask)
    """
    matrix, edge_len = _local_affine_for_edge(p1, p2, outward)
    if edge_len < 1e-6 or samples <= 0:
        return [False] * max(samples, 0), [False] * max(samples, 0)

    raw_shape = _sanitize_geom(raw_shape)
    mod_shape = _sanitize_geom(mod_shape)

    add_geom = _safe_difference(mod_shape, raw_shape)
    sub_geom = _safe_difference(raw_shape, mod_shape)

    local_clip = box(-0.5, -thickness - 0.5, edge_len + 0.5, thickness + 0.5)
    add_local = _safe_intersection(affine_transform(add_geom, matrix), local_clip)
    sub_local = _safe_intersection(affine_transform(sub_geom, matrix), local_clip)

    add_prep = prep(add_local) if not add_local.is_empty else None
    sub_prep = prep(sub_local) if not sub_local.is_empty else None

    tab_probe_y = max(0.2, thickness * 0.5)
    slot_probe_y = -max(0.2, thickness * 0.5)

    tabs = [False] * samples
    slots = [False] * samples
    for i in range(samples):
        x = edge_len * ((i + 0.5) / samples)
        if add_prep is not None:
            tabs[i] = add_prep.intersects(Point(x, tab_probe_y))
        if sub_prep is not None:
            slots[i] = sub_prep.intersects(Point(x, slot_probe_y))

    return tabs, slots


def _evaluate_pair(
    add_a: list[bool],
    sub_a: list[bool],
    add_b: list[bool],
    sub_b: list[bool],
    max_shift: int,
) -> tuple[dict, list[bool], list[bool], list[bool]]:
    """Find best alignment (small shift) and return metrics + mismatch masks."""
    best = None
    best_masks = None

    for shift in range(-max_shift, max_shift + 1):
        add_bs = _shift_mask(add_b, shift)
        sub_bs = _shift_mask(sub_b, shift)

        active_mask = [a or sa or b or sb for a, sa, b, sb in zip(add_a, sub_a, add_bs, sub_bs)]

        mismatch_ab_classic = [m and (x != y) for x, y, m in zip(add_a, sub_bs, active_mask)]
        mismatch_ba_classic = [m and (x != y) for x, y, m in zip(add_bs, sub_a, active_mask)]

        # Preserve-mode joints can appear as "subtractions on both sides".
        # In that case, mating is represented by complementary slot masks.
        not_sub_a = [not x for x in sub_a]
        not_sub_bs = [not x for x in sub_bs]
        mismatch_ab_preserve = [m and (x != y) for x, y, m in zip(not_sub_a, sub_bs, active_mask)]
        mismatch_ba_preserve = [m and (x != y) for x, y, m in zip(not_sub_bs, sub_a, active_mask)]

        mismatch_classic = 0.5 * (
            _masked_xor_ratio(add_a, sub_bs, active_mask)
            + _masked_xor_ratio(add_bs, sub_a, active_mask)
        )
        mismatch_preserve = 0.5 * (
            _masked_xor_ratio(not_sub_a, sub_bs, active_mask)
            + _masked_xor_ratio(not_sub_bs, sub_a, active_mask)
        )

        # If there are effectively no added tabs on either side, prefer preserve mode.
        add_cov = _ratio_true(add_a) + _ratio_true(add_bs)
        if add_cov < 0.02:
            mismatch_ratio = mismatch_preserve
            mismatch_ab_mask = mismatch_ab_preserve
            mismatch_ba_mask = mismatch_ba_preserve
        else:
            if mismatch_preserve < mismatch_classic:
                mismatch_ratio = mismatch_preserve
                mismatch_ab_mask = mismatch_ab_preserve
                mismatch_ba_mask = mismatch_ba_preserve
            else:
                mismatch_ratio = mismatch_classic
                mismatch_ab_mask = mismatch_ab_classic
                mismatch_ba_mask = mismatch_ba_classic

        collision_mask = [x and y for x, y in zip(add_a, add_bs)]
        double_slot_mask = [x and y for x, y in zip(sub_a, sub_bs)]

        collision_ratio = _and_ratio(add_a, add_bs)
        double_slot_ratio = _and_ratio(sub_a, sub_bs)

        key = (mismatch_ratio, collision_ratio, abs(shift))
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "shift": shift,
                "mismatch_ratio": mismatch_ratio,
                "collision_ratio": collision_ratio,
                "double_slot_ratio": double_slot_ratio,
            }
            best_masks = (mismatch_ab_mask, mismatch_ba_mask, collision_mask)

    assert best is not None and best_masks is not None
    return best, best_masks[0], best_masks[1], best_masks[2]


def _debug_mask_svg(
    path: str,
    title: str,
    add_a: list[bool],
    sub_a: list[bool],
    add_b: list[bool],
    sub_b: list[bool],
    mismatch_ab: list[bool],
    mismatch_ba: list[bool],
    collisions: list[bool],
):
    """Write a compact debug SVG visualizing mask alignment."""
    n = len(add_a)
    cell_w = 4
    width = max(320, n * cell_w + 40)
    height = 150

    dwg = svgwrite.Drawing(path, size=(f"{width}px", f"{height}px"), viewBox=f"0 0 {width} {height}")
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))
    dwg.add(dwg.text(title, insert=(8, 12), fill="#111", font_size="10px"))

    def draw_row(y: int, mask: list[bool], color: str, label: str):
        dwg.add(dwg.text(label, insert=(8, y + 8), fill="#444", font_size="8px"))
        x0 = 70
        for i, v in enumerate(mask):
            if v:
                dwg.add(dwg.rect(insert=(x0 + i * cell_w, y), size=(cell_w, 10), fill=color))
        dwg.add(dwg.rect(insert=(x0, y), size=(n * cell_w, 10), fill="none", stroke="#bbb", stroke_width=0.5))

    draw_row(20, add_a, "#111", "A tabs")
    draw_row(34, sub_a, "#666", "A slots")
    draw_row(52, add_b, "#d22", "B tabs")
    draw_row(66, sub_b, "#f66", "B slots")
    draw_row(92, mismatch_ab, "#f9a825", "A tabs vs B slots mismatch")
    draw_row(106, mismatch_ba, "#ef6c00", "B tabs vs A slots mismatch")
    draw_row(124, collisions, "#7b1fa2", "tab-tab overlap")

    dwg.save()


def _collect_shared_joint_edges(
    projections: dict[int, Projection2D],
    shared_edges: list[SharedEdge],
) -> list[dict]:
    joints = []
    for se in shared_edges:
        a = se.face_a_id
        b = se.face_b_id
        if a not in projections or b not in projections:
            continue
        proj_a = projections[a]
        proj_b = projections[b]
        edge_idx_a = _find_matching_edge_index(proj_a, se)
        edge_idx_b = _find_matching_edge_index(proj_b, se)
        if edge_idx_a is None or edge_idx_b is None:
            continue
        joints.append(
            {
                "joint_type": "shared",
                "face_a_id": a,
                "face_b_id": b,
                "p1_a": proj_a.outer_edges_2d[edge_idx_a][0],
                "p2_a": proj_a.outer_edges_2d[edge_idx_a][1],
                "p1_b": proj_b.outer_edges_2d[edge_idx_b][0],
                "p2_b": proj_b.outer_edges_2d[edge_idx_b][1],
            }
        )
    return joints


def _collect_through_slot_edges(
    projections: dict[int, Projection2D],
    shared_edges: list[SharedEdge],
    bottom_id: int,
    faces: list[PlanarFace] | None,
) -> list[dict]:
    if faces is None or bottom_id not in projections:
        return []

    face_map = {f.face_id: f for f in faces}
    bottom_face = face_map.get(bottom_id)
    if bottom_face is None:
        return []

    bottom_adjacent = set()
    for se in shared_edges:
        if se.face_a_id == bottom_id:
            bottom_adjacent.add(se.face_b_id)
        elif se.face_b_id == bottom_id:
            bottom_adjacent.add(se.face_a_id)

    joints = []
    for wall_id, wall_proj in projections.items():
        if wall_id == bottom_id or wall_id in bottom_adjacent:
            continue
        wall_face = face_map.get(wall_id)
        if wall_face is None:
            continue
        endpoints = _find_bottom_edge_endpoints(wall_face, bottom_face)
        if endpoints is None:
            continue
        p_start_3d, p_end_3d = endpoints

        bottom_proj = projections[bottom_id]
        p1_bottom = _project_point(p_start_3d, bottom_proj.origin_3d, bottom_proj.u_axis, bottom_proj.v_axis)
        p2_bottom = _project_point(p_end_3d, bottom_proj.origin_3d, bottom_proj.u_axis, bottom_proj.v_axis)
        p1_wall = _project_point(p_start_3d, wall_proj.origin_3d, wall_proj.u_axis, wall_proj.v_axis)
        p2_wall = _project_point(p_end_3d, wall_proj.origin_3d, wall_proj.u_axis, wall_proj.v_axis)

        joints.append(
            {
                "joint_type": "through_slot",
                "face_a_id": bottom_id,
                "face_b_id": wall_id,
                "p1_a": p1_bottom,
                "p2_a": p2_bottom,
                "p1_b": p1_wall,
                "p2_b": p2_wall,
            }
        )

    return joints


def verify_joint_mesh(
    projections: dict[int, Projection2D],
    modified_polygons: dict[int, list[tuple[float, float]]],
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None,
    shared_edges: list[SharedEdge],
    bottom_id: int,
    thickness: float,
    sample_step: float = 0.25,
    mismatch_tolerance: float = 0.02,
    run_interference: bool = False,
    interference_tolerance: float = 0.01,
    faces: list[PlanarFace] | None = None,
    debug_dir: str | None = None,
) -> VerificationReport:
    """Run seam mesh verification for shared and through-slot joints."""
    raw_shapes = {fid: _shape_from_projection(proj) for fid, proj in projections.items()}
    mod_shapes = {
        fid: _shape_from_modified(fid, modified_polygons, slot_cutouts)
        for fid in projections.keys()
    }

    joints = _collect_shared_joint_edges(projections, shared_edges)
    joints.extend(_collect_through_slot_edges(projections, shared_edges, bottom_id, faces))

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    results: list[JointVerification] = []
    for idx, joint in enumerate(joints):
        a = joint["face_a_id"]
        b = joint["face_b_id"]
        p1_a = joint["p1_a"]
        p2_a = joint["p2_a"]
        p1_b = joint["p1_b"]
        p2_b = joint["p2_b"]

        edge_len_a = math.hypot(p2_a[0] - p1_a[0], p2_a[1] - p1_a[1])
        edge_len_b = math.hypot(p2_b[0] - p1_b[0], p2_b[1] - p1_b[1])
        max_len = max(edge_len_a, edge_len_b)
        samples = max(80, int(max_len / max(sample_step, 0.05)))
        max_shift = max(1, int(round(0.5 / max(sample_step, 0.05))))

        outward_a = _outward_direction(p1_a, p2_a, raw_shapes[a])
        outward_b = _outward_direction(p1_b, p2_b, raw_shapes[b])

        add_a, sub_a = _edge_feature_masks(
            raw_shapes[a], mod_shapes[a], p1_a, p2_a, outward_a, thickness, samples
        )
        add_b_base, sub_b_base = _edge_feature_masks(
            raw_shapes[b], mod_shapes[b], p1_b, p2_b, outward_b, thickness, samples
        )

        # Edge correspondence direction can vary. Evaluate both and keep best.
        best_overall = None
        best_masks = None
        best_reversed = False
        for reversed_b in (False, True):
            add_b = _reverse_mask(add_b_base) if reversed_b else list(add_b_base)
            sub_b = _reverse_mask(sub_b_base) if reversed_b else list(sub_b_base)
            metrics, mismatch_ab, mismatch_ba, collisions = _evaluate_pair(
                add_a, sub_a, add_b, sub_b, max_shift=max_shift
            )
            key = (metrics["mismatch_ratio"], metrics["collision_ratio"], abs(metrics["shift"]))
            if best_overall is None or key < best_overall["key"]:
                best_overall = metrics
                best_masks = {
                    "add_b": add_b,
                    "sub_b": sub_b,
                    "mismatch_ab": mismatch_ab,
                    "mismatch_ba": mismatch_ba,
                    "collisions": collisions,
                }
                best_reversed = reversed_b

        assert best_overall is not None and best_masks is not None

        mismatch_ratio = best_overall["mismatch_ratio"]
        collision_ratio = best_overall["collision_ratio"]
        double_slot_ratio = best_overall["double_slot_ratio"]
        shift = best_overall["shift"]

        add_cov_a = _ratio_true(add_a)
        add_cov_b = _ratio_true(best_masks["add_b"])
        sub_cov_a = _ratio_true(sub_a)
        sub_cov_b = _ratio_true(best_masks["sub_b"])

        has_joint_features = max(add_cov_a, add_cov_b, sub_cov_a, sub_cov_b) > 0.01

        fail_reasons = []
        if has_joint_features and mismatch_ratio > mismatch_tolerance:
            fail_reasons.append(f"mismatch {mismatch_ratio:.3f} > {mismatch_tolerance:.3f}")
        if run_interference and collision_ratio > interference_tolerance:
            fail_reasons.append(f"collision {collision_ratio:.3f} > {interference_tolerance:.3f}")

        passed = len(fail_reasons) == 0
        reason = "ok" if passed else "; ".join(fail_reasons)

        joint_id = f"{joint['joint_type']}_{a}_{b}_{idx}"
        debug_svg = None
        if not passed and debug_dir:
            debug_svg = os.path.join(debug_dir, f"{joint_id}.svg")
            _debug_mask_svg(
                debug_svg,
                title=f"{joint['joint_type']} {a}<->{b}",
                add_a=add_a,
                sub_a=sub_a,
                add_b=best_masks["add_b"],
                sub_b=best_masks["sub_b"],
                mismatch_ab=best_masks["mismatch_ab"],
                mismatch_ba=best_masks["mismatch_ba"],
                collisions=best_masks["collisions"],
            )

        results.append(
            JointVerification(
                joint_id=joint_id,
                joint_type=joint["joint_type"],
                face_a_id=a,
                face_b_id=b,
                samples=samples,
                reversed_b=best_reversed,
                shift_samples=shift,
                mismatch_ratio=mismatch_ratio,
                collision_ratio=collision_ratio,
                double_slot_ratio=double_slot_ratio,
                add_coverage_a=add_cov_a,
                add_coverage_b=add_cov_b,
                sub_coverage_a=sub_cov_a,
                sub_coverage_b=sub_cov_b,
                passed=passed,
                reason=reason,
                debug_svg=debug_svg,
            )
        )

    failed = sum(1 for r in results if not r.passed)
    return VerificationReport(
        total_joints=len(results),
        failed_joints=failed,
        run_interference=run_interference,
        mismatch_tolerance=mismatch_tolerance,
        interference_tolerance=interference_tolerance,
        joints=results,
    )


def write_verification_report(path: str, report: VerificationReport):
    """Write report as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "summary": {
            "total_joints": report.total_joints,
            "failed_joints": report.failed_joints,
            "passed": report.passed,
            "run_interference": report.run_interference,
            "mismatch_tolerance": report.mismatch_tolerance,
            "interference_tolerance": report.interference_tolerance,
        },
        "joints": [asdict(r) for r in report.joints],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
