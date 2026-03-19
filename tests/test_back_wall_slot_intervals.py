from pathlib import Path

import pytest

from lasercut.joints import (
    _BACK_WALL_OUTLINE_CLEARANCE_MM,
    _classify_joint_type,
    _edge_inward_direction,
    _inset_slot_intervals_from_lip,
    _normalize,
    _outer_lip_carrier_intervals,
    _project_edge_to_panel,
    apply_finger_joints,
)
from lasercut.panels import _vec_len, _vec_sub, load_step_panels
from lasercut.exporter import _project_panel
import math


ROOT = Path(__file__).resolve().parents[1]
STEP_FILES = sorted((ROOT / "step_files").glob("*.step"))


def _forbidden_back_wall_seam_segments(step_file: Path) -> list[tuple[str, float]]:
    model = load_step_panels(str(step_file), 3.2)
    seam = next(
        se for se in model.shared_edges if {se.panel_a, se.panel_b} == {"bottom", "back_wall"}
    )

    slot_panel = model.panels["bottom"]
    slot_start, slot_end = _project_edge_to_panel(seam, slot_panel)
    edge_dir = _normalize(_vec_sub(slot_end, slot_start))
    slot_in_plane = _edge_inward_direction(slot_panel, slot_start, slot_end)
    slot_intervals = _inset_slot_intervals_from_lip(
        se=seam,
        slot_panel=slot_panel,
        slot_start=slot_start,
        slot_end=slot_end,
        edge_dir=edge_dir,
        slot_in_plane=slot_in_plane,
        thickness=model.thickness,
        finger_width=20.0,
        start_keepout=0.0,
        end_keepout=0.0,
    )

    jointed = apply_finger_joints(model, finger_width=20.0, kerf=0.0)
    back_wall = jointed.panels["back_wall"]
    projected = _project_panel(back_wall.solid, back_wall.outer_normal, "back_wall")
    assert projected is not None

    seam_start_3d, seam_end_3d = _project_edge_to_panel(seam, back_wall)
    p0 = projected.project_3d(seam_start_3d)
    p1 = projected.project_3d(seam_end_3d)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    seam_len = math.hypot(dx, dy)
    assert seam_len > 1e-6
    ux = dx / seam_len
    uy = dy / seam_len
    nx = -uy
    ny = ux

    def to_tu(pt: tuple[float, float]) -> tuple[float, float]:
        return (
            (pt[0] - p0[0]) * ux + (pt[1] - p0[1]) * uy,
            (pt[0] - p0[0]) * nx + (pt[1] - p0[1]) * ny,
        )

    def outside_slots(t0: float, t1: float) -> bool:
        lo = min(t0, t1)
        hi = max(t0, t1)
        covered = 0.0
        for a, b in slot_intervals:
            covered += max(0.0, min(hi, b) - max(lo, a))
        return (hi - lo) - covered > 0.15

    forbidden: list[tuple[str, float]] = []
    for ring_name, ring in [("outline", projected.outline), *[(f"hole{i}", h) for i, h in enumerate(projected.holes)]]:
        for idx in range(len(ring)):
            a = ring[idx]
            b = ring[(idx + 1) % len(ring)]
            t0, u0 = to_tu(a)
            t1, u1 = to_tu(b)
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg_len < 0.05:
                continue
            if max(abs(u0), abs(u1)) > 0.25:
                continue
            if min(t0, t1) > seam_len + 6.0 or max(t0, t1) < -6.0:
                continue
            if outside_slots(t0, t1):
                forbidden.append((f"{ring_name}:{idx}", seg_len))

    return forbidden


def _tiny_straight_stubs(loop: list[tuple[float, float]], max_len: float = 0.25) -> list[tuple[int, float]]:
    def _is_colinear(
        v1: tuple[float, float],
        v2: tuple[float, float],
        cos_tol: float = 0.995,
    ) -> bool:
        l1 = math.hypot(v1[0], v1[1])
        l2 = math.hypot(v2[0], v2[1])
        if l1 < 1e-9 or l2 < 1e-9:
            return False
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
        return abs(dot) >= cos_tol

    stubs: list[tuple[int, float]] = []
    for idx in range(len(loop)):
        prev_i = (idx - 1) % len(loop)
        next_i = (idx + 1) % len(loop)
        next2_i = (idx + 2) % len(loop)
        a = loop[idx]
        b = loop[next_i]
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg_len >= max_len:
            continue

        v_prev = (a[0] - loop[prev_i][0], a[1] - loop[prev_i][1])
        v_short = (b[0] - a[0], b[1] - a[1])
        v_next = (loop[next2_i][0] - b[0], loop[next2_i][1] - b[1])
        if _is_colinear(v_prev, v_short) or _is_colinear(v_short, v_next):
            stubs.append((idx, seg_len))

    return stubs


@pytest.mark.parametrize("step_file", STEP_FILES, ids=lambda path: path.name)
def test_bottom_back_wall_slots_follow_irregular_lip(step_file: Path) -> None:
    model = load_step_panels(str(step_file), 3.2)
    seam = next(
        se for se in model.shared_edges if {se.panel_a, se.panel_b} == {"bottom", "back_wall"}
    )

    joint_type, slot_panel_name = _classify_joint_type(seam, model.panels)
    assert joint_type == "through_slot"
    assert slot_panel_name == "bottom"

    slot_panel = model.panels[slot_panel_name]
    slot_start, slot_end = _project_edge_to_panel(seam, slot_panel)
    edge_dir = _normalize(_vec_sub(slot_end, slot_start))
    assert _vec_len(_vec_sub(slot_end, slot_start)) > 1e-6

    slot_in_plane = _edge_inward_direction(slot_panel, slot_start, slot_end)
    carrier_runs = _outer_lip_carrier_intervals(
        slot_panel=slot_panel,
        slot_start=slot_start,
        slot_end=slot_end,
        edge_dir=edge_dir,
        slot_in_plane=slot_in_plane,
        thickness=model.thickness,
    )
    assert carrier_runs, f"{step_file.name}: expected outer lip carrier runs on bottom"

    slot_intervals = _inset_slot_intervals_from_lip(
        se=seam,
        slot_panel=slot_panel,
        slot_start=slot_start,
        slot_end=slot_end,
        edge_dir=edge_dir,
        slot_in_plane=slot_in_plane,
        thickness=model.thickness,
        finger_width=20.0,
        start_keepout=0.0,
        end_keepout=0.0,
    )
    assert slot_intervals, f"{step_file.name}: expected segmented bottom/back-wall slots"

    clearance_mm = _BACK_WALL_OUTLINE_CLEARANCE_MM - 0.05
    for lo, hi in slot_intervals:
        assert hi > lo
        assert any(
            carrier_lo + clearance_mm <= lo and hi <= carrier_hi - clearance_mm
            for carrier_lo, carrier_hi in carrier_runs
        ), f"{step_file.name}: slot interval {(lo, hi)} violates 1 mm edge clearance"


@pytest.mark.parametrize("step_file", STEP_FILES, ids=lambda path: path.name)
def test_back_wall_recess_overruns_joint_ends(step_file: Path) -> None:
    model = load_step_panels(str(step_file), 3.2)

    jointed = apply_finger_joints(model, finger_width=20.0, kerf=0.0)
    seam = next(
        se for se in jointed.shared_edges if {se.panel_a, se.panel_b} == {"bottom", "back_wall"}
    )
    back_wall = jointed.panels["back_wall"]
    projected = _project_panel(back_wall.solid, back_wall.outer_normal, "back_wall")
    assert projected is not None

    seam_start_3d, seam_end_3d = _project_edge_to_panel(seam, back_wall)
    p0 = projected.project_3d(seam_start_3d)
    p1 = projected.project_3d(seam_end_3d)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    seam_len = math.hypot(dx, dy)
    assert seam_len > 1e-6
    ux = dx / seam_len
    uy = dy / seam_len
    nx = -uy
    ny = ux

    seam_band = []
    for idx, (x, y) in enumerate(projected.outline):
        t = (x - p0[0]) * ux + (y - p0[1]) * uy
        u = (x - p0[0]) * nx + (y - p0[1]) * ny
        if -6.0 <= t <= seam_len + 6.0 and -1.0 <= u <= 6.0:
            seam_band.append((t, u, idx))

    assert any(t <= -2.5 and u >= 3.0 for t, u, _ in seam_band), (
        f"{step_file.name}: expected back-wall recess to overrun seam start"
    )
    assert any(t >= seam_len + 2.5 and u >= 3.0 for t, u, _ in seam_band), (
        f"{step_file.name}: expected back-wall recess to overrun seam end"
    )


@pytest.mark.parametrize("step_file", STEP_FILES, ids=lambda path: path.name)
def test_back_wall_bottom_seam_has_no_ghost_contacts(step_file: Path) -> None:
    forbidden = _forbidden_back_wall_seam_segments(step_file)
    assert not forbidden, f"{step_file.name}: unexpected seam-line residues {forbidden}"


@pytest.mark.parametrize("step_file", STEP_FILES, ids=lambda path: path.name)
def test_projected_loops_have_no_tiny_straight_stubs(step_file: Path) -> None:
    model = load_step_panels(str(step_file), 3.2)
    jointed = apply_finger_joints(model, finger_width=20.0, kerf=0.0)

    offenders: list[str] = []
    for name, panel in jointed.panels.items():
        projected = _project_panel(panel.solid, panel.outer_normal, name)
        assert projected is not None

        outline_stubs = _tiny_straight_stubs(projected.outline)
        if outline_stubs:
            offenders.append(f"{name}:outline:{outline_stubs}")

        for hole_idx, hole in enumerate(projected.holes):
            hole_stubs = _tiny_straight_stubs(hole)
            if hole_stubs:
                offenders.append(f"{name}:hole{hole_idx}:{hole_stubs}")

    assert not offenders, f"{step_file.name}: tiny straight stubs remained {offenders}"
