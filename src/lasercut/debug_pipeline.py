"""Generate stepwise debug artifacts for the lasercut pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace

import cadquery as cq

from lasercut.exporter import (
    Affine2D,
    _project_panel,
    _compute_unfolded_layout,
    _translate_pts,
    svgwrite,
)
from lasercut.joints import (
    _classify_joint_type,
    _is_edge_on_boundary,
    _project_edge_to_panel,
    _seam_panel_angle_deg,
    _should_use_living_hinge,
    apply_finger_joints,
)
from lasercut.panels import (
    BinModel,
    Panel,
    _compute_in_plane_dims,
    _edge_overlap_length,
    _extract_outer_wire_edges,
    _find_shared_edges,
    _point_to_line_dist,
    _thicken_face_inward,
    _vec_dot,
    _vec_sub,
    _vec_len,
    load_step_panels,
)


def _num_token(value: float) -> str:
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return s or "0"


@dataclass
class PipelineStage:
    id: str
    title: str
    subtitle: str
    model: BinModel
    input_stage: str | None
    reference_model: BinModel | None = None
    seam_infos: list[dict] | None = None
    notes: list[str] | None = None


def _clone_panel_with_solid(panel: Panel, solid: cq.Shape) -> Panel:
    return replace(panel, solid=solid)


def _build_raw_model(model: BinModel) -> BinModel:
    raw_panels: dict[str, Panel] = {}
    for name, panel in model.panels.items():
        raw_panels[name] = _clone_panel_with_solid(
            panel,
            panel.reference_solid if panel.reference_solid is not None else panel.solid,
        )
    return BinModel(
        panels=raw_panels,
        shared_edges=model.shared_edges,
        thickness=model.thickness,
        source_solid=model.source_solid,
        living_hinge_seams=model.living_hinge_seams,
    )


def _find_inner_face_for_panel(model: BinModel, panel: Panel) -> tuple[cq.Face, tuple[float, float, float]] | None:
    if model.source_solid is None or panel.outer_face is None:
        return None

    oc = panel.outer_face.Center()
    oc_t = (oc.x, oc.y, oc.z)
    n = panel.outer_normal
    outer_area = panel.outer_face.Area()

    best_face: cq.Face | None = None
    best_normal: tuple[float, float, float] | None = None
    best_score: tuple[float, float, float] | None = None

    for face in model.source_solid.Faces():
        try:
            fc = face.Center()
            fn = face.normalAt(fc)
        except Exception:
            continue
        fn_t = (fn.x, fn.y, fn.z)
        if _vec_dot(fn_t, n) > -0.99:
            continue

        dc = (fc.x - oc_t[0], fc.y - oc_t[1], fc.z - oc_t[2])
        sep_signed = _vec_dot(dc, n)
        sep = abs(sep_signed)
        if sep < 0.2:
            continue

        lateral = (
            dc[0] - n[0] * sep_signed,
            dc[1] - n[1] * sep_signed,
            dc[2] - n[2] * sep_signed,
        )
        lateral_dist = _vec_len(lateral)
        area_ratio = min(face.Area(), outer_area) / max(face.Area(), outer_area)

        score = (lateral_dist, 1.0 - area_ratio, sep)
        if best_score is None or score < best_score:
            best_score = score
            best_face = face
            best_normal = fn_t

    if best_face is None or best_normal is None:
        return None
    return best_face, best_normal


def _build_inner_raw_model(model: BinModel) -> BinModel:
    inner_panels: dict[str, Panel] = {}
    for name, panel in model.panels.items():
        # For the bottom panel, the outer/lower face is the more useful raw
        # reference because it preserves the true outer footprint. The inner
        # face is still useful for walls, but it hides exactly the bottom-edge
        # geometry we want to inspect here.
        if name == "bottom":
            inner_panels[name] = _clone_panel_with_solid(
                panel,
                panel.reference_solid if panel.reference_solid is not None else panel.solid,
            )
            continue

        found = _find_inner_face_for_panel(model, panel)
        if found is None:
            inner_panels[name] = _clone_panel_with_solid(
                panel,
                panel.reference_solid if panel.reference_solid is not None else panel.solid,
            )
            continue

        inner_face, inner_normal = found
        inner_solid = _thicken_face_inward(inner_face, inner_normal, model.thickness)
        inner_edges = _extract_outer_wire_edges(inner_face)
        width, height = _compute_in_plane_dims(inner_edges, inner_normal)
        inner_panels[name] = replace(
            panel,
            solid=inner_solid,
            outer_normal=inner_normal,
            width=width,
            height=height,
            outer_face=inner_face,
            outer_edges=inner_edges,
            reference_solid=panel.reference_solid,
            export_additions=[],
            debug_cut_lines=[],
        )

    return BinModel(
        panels=inner_panels,
        shared_edges=model.shared_edges,
        thickness=model.thickness,
        source_solid=model.source_solid,
        living_hinge_seams=model.living_hinge_seams,
    )


def _build_working_model(model: BinModel) -> BinModel:
    """Build the semantically cleaned working geometry used for seam reasoning.

    For the current single-solid experiments this means:
    - bottom stays on the useful outer/lower face
    - walls/gussets use the opposite inner face where possible
    - shared edges are recomputed from that geometry
    """
    panels_model = _build_inner_raw_model(model)
    return BinModel(
        panels=panels_model.panels,
        shared_edges=_find_shared_edges(panels_model.panels),
        thickness=panels_model.thickness,
        source_solid=panels_model.source_solid,
        living_hinge_seams=panels_model.living_hinge_seams,
    )


def _build_unfolded_scene(model: BinModel):
    panel_map = {}
    for name, panel in model.panels.items():
        p2d = _project_panel(panel.solid, panel.outer_normal, name)
        if p2d is not None:
            panel_map[name] = p2d
    if not panel_map:
        raise ValueError("No outlines could be projected for debug stage")

    placed = _compute_unfolded_layout(model, panel_map, gap=4.0)
    if not placed:
        raise ValueError("No panels could be placed for debug stage")

    all_x = [p[0] for _, pts, _, _ in placed for p in pts]
    all_y = [p[1] for _, pts, _, _ in placed for p in pts]
    min_x = min(all_x)
    min_y = min(all_y)
    shift_x = -min_x + 5.0 if min_x < 0 else 0.0
    shift_y = -min_y + 5.0 if min_y < 0 else 0.0
    if shift_x > 0 or shift_y > 0:
        placed = [
            (
                name,
                _translate_pts(pts, shift_x, shift_y),
                [_translate_pts(hole, shift_x, shift_y) for hole in holes],
                Affine2D.from_translation(shift_x, shift_y).compose(xform),
            )
            for name, pts, holes, xform in placed
        ]

    all_x = [p[0] for _, pts, _, _ in placed for p in pts]
    all_y = [p[1] for _, pts, _, _ in placed for p in pts]
    total_w = max(all_x) + 5.0
    total_h = max(all_y) + 5.0
    placed_by_name = {
        name: {"outline": pts, "holes": holes, "xform": xform}
        for name, pts, holes, xform in placed
    }
    return panel_map, placed_by_name, total_w, total_h


def _draw_model_stage(
    model: BinModel,
    output_path: str,
    *,
    title: str,
    subtitle: str | None = None,
    reference_model: BinModel | None = None,
    seam_infos: list[dict] | None = None,
) -> None:
    if svgwrite is None:
        raise ImportError("svgwrite is required for debug pipeline exports")

    panel_map, placed_by_name, total_w, total_h = _build_unfolded_scene(model)

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{total_w}mm", f"{total_h}mm"),
        viewBox=f"0 0 {total_w} {total_h}",
    )
    dwg.add(dwg.rect(insert=(0, 0), size=(total_w, total_h), fill="#FFFFFF", stroke="none"))

    dwg.add(
        dwg.text(
            title,
            insert=(4, 4),
            text_anchor="start",
            dominant_baseline="hanging",
            font_size="5",
            font_weight="bold",
            fill="#202020",
        )
    )
    if subtitle:
        dwg.add(
            dwg.text(
                subtitle,
                insert=(4, 10),
                text_anchor="start",
                dominant_baseline="hanging",
                font_size="3.3",
                fill="#666666",
            )
        )

    for name, placed in placed_by_name.items():
        pts = placed["outline"]
        holes = placed["holes"]
        xform = placed["xform"]
        d_parts = [f"M {pts[0][0]:.4f},{pts[0][1]:.4f}"]
        for p in pts[1:]:
            d_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
        d_parts.append("Z")

        if reference_model is not None and name in reference_model.panels and name in panel_map:
            ref_panel = reference_model.panels[name]
            ref_p2d = _project_panel(ref_panel.solid, ref_panel.outer_normal, name, frame=panel_map[name])
            if ref_p2d is not None and ref_p2d.outline:
                ref_outline = xform.apply_pts(ref_p2d.outline)
                r_parts = [f"M {ref_outline[0][0]:.4f},{ref_outline[0][1]:.4f}"]
                for p in ref_outline[1:]:
                    r_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
                r_parts.append("Z")
                dwg.add(
                    dwg.path(
                        d=" ".join(r_parts),
                        stroke="#1BAA5C",
                        stroke_width="0.4",
                        stroke_dasharray="2.0,1.6",
                        fill="none",
                        opacity="0.9",
                    )
                )

        dwg.add(dwg.path(d=" ".join(d_parts), stroke="#000000", stroke_width="0.5", fill="none"))

        current_panel = model.panels.get(name)
        current_p2d = panel_map.get(name)
        if current_panel is not None and current_p2d is not None:
            for p0_3d, p1_3d in current_panel.debug_cut_lines:
                p0 = xform.apply(*current_p2d.project_3d(p0_3d))
                p1 = xform.apply(*current_p2d.project_3d(p1_3d))
                dwg.add(
                    dwg.path(
                        d=f"M {p0[0]:.4f},{p0[1]:.4f} L {p1[0]:.4f},{p1[1]:.4f}",
                        stroke="#D94A4A",
                        stroke_width="0.45",
                        stroke_dasharray="1.8,1.4",
                        fill="none",
                        opacity="0.95",
                    )
                )

        for hole in holes:
            if len(hole) == 2:
                dwg.add(
                    dwg.path(
                        d=f"M {hole[0][0]:.4f},{hole[0][1]:.4f} L {hole[1][0]:.4f},{hole[1][1]:.4f}",
                        stroke="#000000",
                        stroke_width="0.5",
                        fill="none",
                    )
                )
                continue
            if len(hole) >= 3:
                h_parts = [f"M {hole[0][0]:.4f},{hole[0][1]:.4f}"]
                for p in hole[1:]:
                    h_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
                h_parts.append("Z")
                dwg.add(dwg.path(d=" ".join(h_parts), stroke="#000000", stroke_width="0.5", fill="none"))

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        dwg.add(
            dwg.text(
                name,
                insert=(cx, cy),
                text_anchor="middle",
                dominant_baseline="central",
                font_size="4",
                fill="#222222",
            )
        )

    if seam_infos:
        seam_palette = [
            "#246BFD",
            "#D94A4A",
            "#1BAA5C",
            "#B12A90",
            "#F08C00",
            "#0E7490",
            "#7C3AED",
            "#CA8A04",
            "#BE185D",
            "#059669",
        ]
        seam_color_by_id = {
            info["id"]: seam_palette[idx % len(seam_palette)]
            for idx, info in enumerate(seam_infos)
        }
        legend_y = 16.0 if subtitle else 10.0
        legend = [
            ("finger", None),
            ("through_slot", "1.2,0.9"),
            ("living_hinge", "2.0,1.0"),
        ]
        lx = 4.0
        for label, dash in legend:
            line_kwargs = {
                "start": (lx, legend_y),
                "end": (lx + 8, legend_y),
                "stroke": "#444444",
                "stroke_width": 0.9,
            }
            if dash is not None:
                line_kwargs["stroke_dasharray"] = dash
            dwg.add(dwg.line(**line_kwargs))
            dwg.add(
                dwg.text(
                    label,
                    insert=(lx + 10, legend_y),
                    text_anchor="start",
                    dominant_baseline="central",
                    font_size="3.2",
                    fill="#444444",
                )
            )
            lx += 34

        list_y = legend_y + 6.0
        for info in seam_infos:
            kind = info["kind"]
            color = seam_color_by_id[info["id"]]
            pair_text = (
                f"{info['id']}  {info['panel_a']} <-> {info['panel_b']}  "
                f"[{kind}, {info['edge_length']:.1f} mm]"
            )
            dwg.add(
                dwg.text(
                    pair_text,
                    insert=(4, list_y),
                    text_anchor="start",
                    dominant_baseline="central",
                    font_size="3.0",
                    fill=color,
                )
            )
            list_y += 4.0

        for info in seam_infos:
            seam = info["seam"]
            kind = info["kind"]
            color = seam_color_by_id[info["id"]]
            for side_idx, (panel_name, seg_key) in enumerate(
                ((seam.panel_a, "display_segment_a"), (seam.panel_b, "display_segment_b"))
            ):
                if panel_name not in placed_by_name or panel_name not in panel_map:
                    continue
                placed = placed_by_name[panel_name]
                xform = placed["xform"]
                p2d = panel_map[panel_name]
                seg = info.get(seg_key)
                if seg is not None:
                    p0 = xform.apply(*p2d.project_3d(seg[0]))
                    p1 = xform.apply(*p2d.project_3d(seg[1]))
                else:
                    p0 = xform.apply(*p2d.project_3d(seam.start_3d))
                    p1 = xform.apply(*p2d.project_3d(seam.end_3d))
                line_kwargs = {
                    "start": p0,
                    "end": p1,
                    "stroke": color,
                    "stroke_width": 0.9,
                    "opacity": "0.9",
                }
                if kind == "through_slot":
                    line_kwargs["stroke_dasharray"] = "1.2,0.9"
                elif kind == "living_hinge":
                    line_kwargs["stroke_dasharray"] = "2.0,1.0"
                dwg.add(dwg.line(**line_kwargs))
                mx = (p0[0] + p1[0]) / 2.0
                my = (p0[1] + p1[1]) / 2.0
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                seg_len = (dx * dx + dy * dy) ** 0.5
                if seg_len > 1e-6:
                    nx = -dy / seg_len
                    ny = dx / seg_len
                else:
                    nx = 0.0
                    ny = -1.0
                offset = 2.0 if side_idx == 0 else -2.0
                dwg.add(
                    dwg.text(
                        f"{info['id']}{'A' if side_idx == 0 else 'B'}",
                        insert=(mx + nx * offset, my + ny * offset),
                        text_anchor="middle",
                        dominant_baseline="central",
                        font_size="3.0",
                        fill=color,
                    )
                )

    dwg.save()


def _bbox_dict(shape: cq.Shape) -> dict[str, float]:
    bb = shape.BoundingBox()
    return {
        "xmin": float(bb.xmin),
        "ymin": float(bb.ymin),
        "zmin": float(bb.zmin),
        "xmax": float(bb.xmax),
        "ymax": float(bb.ymax),
        "zmax": float(bb.zmax),
        "xlen": float(bb.xlen),
        "ylen": float(bb.ylen),
        "zlen": float(bb.zlen),
    }


def _best_boundary_segment_for_source_seam(
    se,
    panel: Panel,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float] | None:
    se_start, se_end = _project_edge_to_panel(se, panel)
    se_vec = _vec_sub(se_end, se_start)
    se_len = _vec_len(se_vec)
    if se_len < 1e-6:
        return None
    se_dir = (se_vec[0] / se_len, se_vec[1] / se_len, se_vec[2] / se_len)

    best = None
    best_score = None
    for edge in panel.outer_edges:
        pe_vec = _vec_sub(edge[1], edge[0])
        pe_len = _vec_len(pe_vec)
        if pe_len < 0.1:
            continue
        pe_dir = (pe_vec[0] / pe_len, pe_vec[1] / pe_len, pe_vec[2] / pe_len)
        parallel = abs(_vec_dot(se_dir, pe_dir))
        if parallel < 0.94:
            continue

        d0 = _point_to_line_dist(edge[0], se_start, se_end)
        d1 = _point_to_line_dist(edge[1], se_start, se_end)
        worst_dist = max(d0, d1)
        overlap = _edge_overlap_length(se_start, se_end, edge[0], edge[1])
        score = (worst_dist, -overlap, 1.0 - parallel)
        if best_score is None or score < best_score:
            best_score = score
            if _vec_dot(pe_vec, se_vec) < 0:
                best = (edge[1], edge[0], worst_dist)
            else:
                best = (edge[0], edge[1], worst_dist)

    return best


def _seam_debug_info(
    classification_model: BinModel,
    display_model: BinModel,
    living_hinge_angle: float,
) -> list[dict]:
    infos: list[dict] = []
    for idx, se in enumerate(classification_model.shared_edges, start=1):
        joint_type, slot_panel = _classify_joint_type(se, classification_model.panels)
        angle = _seam_panel_angle_deg(se, classification_model.panels)
        is_hinge = joint_type == "finger" and _should_use_living_hinge(
            se,
            classification_model.panels,
            living_hinge_angle,
        )
        kind = "living_hinge" if is_hinge else joint_type
        seg_a = _best_boundary_segment_for_source_seam(se, display_model.panels[se.panel_a])
        seg_b = _best_boundary_segment_for_source_seam(se, display_model.panels[se.panel_b])
        infos.append(
            {
                "id": f"S{idx}",
                "pair_name": f"{se.panel_a}<->{se.panel_b}",
                "kind": kind,
                "slot_panel": slot_panel,
                "panel_a": se.panel_a,
                "panel_b": se.panel_b,
                "edge_length": round(se.edge_length, 4),
                "angle_deg": round(angle, 3),
                "boundary_a": _is_edge_on_boundary(se, classification_model.panels[se.panel_a]),
                "boundary_b": _is_edge_on_boundary(se, classification_model.panels[se.panel_b]),
                "display_segment_a": seg_a,
                "display_segment_b": seg_b,
                "seam": se,
            }
        )
    return infos


def _json_report(
    step_file: str,
    stages: list[PipelineStage],
    seam_infos: list[dict],
    output_dir: str,
    thickness: float,
    finger_width: float,
    kerf: float,
    living_hinge_angle: float,
) -> dict:
    final_stage = stages[-1]
    model = final_stage.model
    panels = []
    for name, panel in model.panels.items():
        panels.append(
            {
                "name": name,
                "bbox": _bbox_dict(panel.solid),
                "has_reference_solid": panel.reference_solid is not None,
                "has_cleanup_plane": panel.cleanup_plane_point is not None,
                "export_additions": len(panel.export_additions),
                "debug_cut_lines": len(panel.debug_cut_lines),
            }
        )
    seams = []
    for info in seam_infos:
        seam = dict(info)
        seam.pop("seam", None)
        for key in ("display_segment_a", "display_segment_b"):
            seg = seam.get(key)
            if seg is not None:
                seam[key] = [
                    [float(v) for v in seg[0]],
                    [float(v) for v in seg[1]],
                    float(seg[2]),
                ]
        seams.append(seam)
    return {
        "step_file": step_file,
        "thickness": thickness,
        "finger_width": finger_width,
        "kerf": kerf,
        "living_hinge_angle": living_hinge_angle,
        "output_dir": output_dir,
        "panel_count": len(model.panels),
        "shared_edge_count": len(model.shared_edges),
        "living_hinge_count": sum(1 for info in seam_infos if info["kind"] == "living_hinge"),
        "through_slot_count": sum(1 for info in seam_infos if info["kind"] == "through_slot"),
        "stages": [
            {
                "id": stage.id,
                "title": stage.title,
                "artifact": f"{stage.id}.svg",
                "input_stage": stage.input_stage,
                "panel_count": len(stage.model.panels),
                "shared_edge_count": len(stage.model.shared_edges),
                "notes": stage.notes or [],
            }
            for stage in stages
        ],
        "panels": panels,
        "seams": seams,
        "final_panel_bboxes": {
            name: _bbox_dict(panel.solid) for name, panel in final_stage.model.panels.items()
        },
        "artifacts": {
            "00_raw_extract": "00_raw_extract.svg",
            "01_raw_vs_clean": "01_raw_vs_clean.svg",
            "02_clean_geometry": "02_clean_geometry.svg",
            "03_seams": "03_seams.svg",
            "04_joint_application": "04_joint_application.svg",
            "X_current_generator_comparison": "X_current_generator_comparison.svg",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stepwise debug artifacts for lasercut models")
    parser.add_argument("step_file", help="Path to STEP file")
    parser.add_argument("--thickness", type=float, default=3.2, help="Material thickness in mm")
    parser.add_argument("--finger-width", type=float, default=20.0, help="Target finger width in mm")
    parser.add_argument("--kerf", type=float, default=0.0, help="Kerf compensation in mm")
    parser.add_argument(
        "--living-hinge-angle",
        type=float,
        default=45.0,
        help="Living-hinge threshold angle in degrees (<=0 disables)",
    )
    parser.add_argument("--output", default="output/debug-pipeline", help="Output root directory")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.step_file))[0]
    folder_name = f"{name}-t{_num_token(args.thickness)}-k{_num_token(args.kerf)}"
    output_dir = os.path.join(args.output, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    for stale_name in [
        "00_raw_extract.svg",
        "01_raw_vs_clean.svg",
        "02_clean_geometry.svg",
        "03_seams.svg",
        "04_final_joints.svg",
        "04_current_generator_joints.svg",
        "04_joint_application.svg",
        "05_pipeline_attempt_joints.svg",
        "X_current_generator_comparison.svg",
        "report.json",
    ]:
        stale_path = os.path.join(output_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    source_model = load_step_panels(args.step_file, args.thickness)
    outer_raw_model = _build_raw_model(source_model)
    raw_model = _build_inner_raw_model(source_model)
    clean_model = _build_working_model(source_model)
    seam_infos = _seam_debug_info(clean_model, clean_model, args.living_hinge_angle)
    joint_input_model = BinModel(
        panels=clean_model.panels,
        shared_edges=clean_model.shared_edges,
        thickness=clean_model.thickness,
        source_solid=None,
    )
    joint_application_model = apply_finger_joints(
        joint_input_model,
        args.finger_width,
        kerf=args.kerf,
        living_hinge_angle_threshold_deg=args.living_hinge_angle,
    )
    current_generator_model = apply_finger_joints(
        source_model,
        args.finger_width,
        kerf=args.kerf,
        living_hinge_angle_threshold_deg=args.living_hinge_angle,
    )

    stages = [
        PipelineStage(
            id="00_raw_extract",
            title="00 Raw Extract",
            subtitle="Black = inner-face raw extract for walls, bottom keeps outer face; green dashed = outer-face raw extract",
            model=raw_model,
            input_stage=None,
            reference_model=outer_raw_model,
            notes=["Initial extraction from STEP before any cleanup or seam recomputation."],
        ),
        PipelineStage(
            id="01_raw_vs_clean",
            title="01 Raw vs Clean",
            subtitle="Black = working geometry, green dashed = raw extract basis",
            model=clean_model,
            input_stage="00_raw_extract",
            reference_model=raw_model,
            notes=["Uses stage 00 panel solids as input and recomputes shared edges from the extracted panel shapes."],
        ),
        PipelineStage(
            id="02_clean_geometry",
            title="02 Clean Geometry",
            subtitle="Black = cleaned working geometry that will feed the numbered stages below",
            model=clean_model,
            input_stage="01_raw_vs_clean",
            notes=["Canonical pre-joint stage used for seam display and joint application."],
        ),
        PipelineStage(
            id="03_seams",
            title="03 Seam Classification",
            subtitle="Seams classified only from stage 02 geometry and drawn on the same stage 02 outlines",
            model=clean_model,
            input_stage="02_clean_geometry",
            seam_infos=seam_infos,
            notes=["No fallback to source-model seams in numbered stages."],
        ),
        PipelineStage(
            id="04_joint_application",
            title="04 Joint Application",
            subtitle="Black = joints applied directly to stage 03 geometry; green dashed = stage 02 clean geometry",
            model=joint_application_model,
            input_stage="03_seams",
            reference_model=clean_model,
            notes=["This is the true sequential continuation of stage 03.", "If this looks wrong, the break is between stage 03 seam logic and joint application."],
        ),
    ]

    for stage in stages:
        _draw_model_stage(
            stage.model,
            os.path.join(output_dir, f"{stage.id}.svg"),
            title=stage.title,
            subtitle=stage.subtitle,
            reference_model=stage.reference_model,
            seam_infos=stage.seam_infos,
        )

    _draw_model_stage(
        current_generator_model,
        os.path.join(output_dir, "X_current_generator_comparison.svg"),
        title="X Current Generator Comparison",
        subtitle="Black = current production output from source geometry, green dashed = original source pre-joint geometry",
        reference_model=source_model,
    )

    report = _json_report(
        args.step_file,
        stages,
        seam_infos,
        output_dir,
        args.thickness,
        args.finger_width,
        args.kerf,
        args.living_hinge_angle,
    )
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Debug pipeline written to {output_dir}")


if __name__ == "__main__":
    main()
