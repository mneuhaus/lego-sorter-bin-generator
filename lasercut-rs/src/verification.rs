use std::collections::HashMap;

use geo::algorithm::area::Area;
use geo::algorithm::contains::Contains;
use geo::{Coord, LineString, Polygon as GeoPolygon};

use crate::finger_joints::{find_bottom_edge_endpoints, find_matching_edge_index};
use crate::math_utils::*;
use crate::types::*;

fn polygon_to_geo(outer: &[Vec2], holes: &[Vec<Vec2>]) -> GeoPolygon<f64> {
    if outer.len() < 3 {
        return GeoPolygon::new(LineString::new(vec![]), vec![]);
    }
    let mut coords: Vec<Coord<f64>> = outer.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
    if coords.first() != coords.last() {
        coords.push(coords[0]);
    }
    let exterior = LineString::new(coords);

    let interiors: Vec<LineString<f64>> = holes
        .iter()
        .filter(|h| h.len() >= 3)
        .map(|h| {
            let mut cs: Vec<Coord<f64>> = h.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
            if cs.first() != cs.last() {
                cs.push(cs[0]);
            }
            LineString::new(cs)
        })
        .collect();

    GeoPolygon::new(exterior, interiors)
}

fn ratio_true(mask: &[bool]) -> f64 {
    if mask.is_empty() {
        return 0.0;
    }
    mask.iter().filter(|&&v| v).count() as f64 / mask.len() as f64
}

fn and_ratio(a: &[bool], b: &[bool]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .filter(|(&x, &y)| x && y)
        .count() as f64
        / a.len() as f64
}

fn masked_xor_ratio(a: &[bool], b: &[bool], mask: &[bool]) -> f64 {
    let mut n = 0;
    let mut bad = 0;
    for ((&x, &y), &m) in a.iter().zip(b.iter()).zip(mask.iter()) {
        if !m {
            continue;
        }
        n += 1;
        if x != y {
            bad += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        bad as f64 / n as f64
    }
}

fn reverse_mask(mask: &[bool]) -> Vec<bool> {
    mask.iter().rev().copied().collect()
}

fn shift_mask(mask: &[bool], shift: i32) -> Vec<bool> {
    let n = mask.len();
    if n == 0 || shift == 0 {
        return mask.to_vec();
    }
    let mut out = vec![false; n];
    if shift > 0 {
        let s = shift as usize;
        if s < n {
            out[s..].copy_from_slice(&mask[..n - s]);
        }
    } else {
        let s = (-shift) as usize;
        if s < n {
            out[..n - s].copy_from_slice(&mask[s..]);
        }
    }
    out
}

fn local_affine_for_edge(
    p1: Vec2,
    p2: Vec2,
    outward: Vec2,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let edge_len = (dx * dx + dy * dy).sqrt();
    if edge_len < 1e-9 {
        return (1.0, 0.0, 0.0, 1.0, -p1[0], -p1[1], 0.0);
    }

    let ux = dx / edge_len;
    let uy = dy / edge_len;
    let vx = outward[0];
    let vy = outward[1];

    let xoff = -(p1[0] * ux + p1[1] * uy);
    let yoff = -(p1[0] * vx + p1[1] * vy);
    (ux, uy, vx, vy, xoff, yoff, edge_len)
}

fn transform_point(x: f64, y: f64, ux: f64, uy: f64, vx: f64, vy: f64, xoff: f64, yoff: f64) -> (f64, f64) {
    (x * ux + y * uy + xoff, x * vx + y * vy + yoff)
}

fn outward_direction_for_verify(
    p1: Vec2,
    p2: Vec2,
    polygon: &GeoPolygon<f64>,
) -> Vec2 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let length = (dx * dx + dy * dy).sqrt();
    if length < 1e-12 {
        return [0.0, 1.0];
    }

    let n1 = [-dy / length, dx / length];
    let n2 = [dy / length, -dx / length];
    let mid = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0];
    let probe = length.min(1.0).max(0.2) * 0.01;

    let test1 = Coord {
        x: mid[0] + n1[0] * probe,
        y: mid[1] + n1[1] * probe,
    };
    let test2 = Coord {
        x: mid[0] + n2[0] * probe,
        y: mid[1] + n2[1] * probe,
    };

    let inside1 = polygon.contains(&test1);
    let inside2 = polygon.contains(&test2);

    if inside1 && !inside2 {
        return n2;
    }
    if inside2 && !inside1 {
        return n1;
    }

    // Fallback centroid
    let ext_coords: Vec<_> = polygon.exterior().coords().collect();
    let n = ext_coords.len() as f64;
    let cx: f64 = ext_coords.iter().map(|c| c.x).sum::<f64>() / n;
    let cy: f64 = ext_coords.iter().map(|c| c.y).sum::<f64>() / n;

    let d1 = (mid[0] + n1[0] - cx).powi(2) + (mid[1] + n1[1] - cy).powi(2);
    let d2 = (mid[0] + n2[0] - cx).powi(2) + (mid[1] + n2[1] - cy).powi(2);
    if d1 > d2 { n1 } else { n2 }
}

fn edge_feature_masks(
    raw_shape: &GeoPolygon<f64>,
    mod_shape: &GeoPolygon<f64>,
    p1: Vec2,
    p2: Vec2,
    outward: Vec2,
    thickness: f64,
    samples: usize,
) -> (Vec<bool>, Vec<bool>) {
    let (_, _, _, _, _, _, edge_len) = local_affine_for_edge(p1, p2, outward);
    if edge_len < 1e-6 || samples == 0 {
        return (vec![false; samples], vec![false; samples]);
    }

    let tab_probe_y: f64 = (0.2_f64).max(thickness * 0.5);

    let mut tabs = vec![false; samples];
    let mut slots = vec![false; samples];

    for i in 0..samples {
        let t = (i as f64 + 0.5) / samples as f64;
        let base = lerp2(p1, p2, t);

        // Tab probe: outward from edge
        let tab_pt = Coord {
            x: base[0] + outward[0] * tab_probe_y,
            y: base[1] + outward[1] * tab_probe_y,
        };
        // Slot probe: inward from edge
        let slot_pt = Coord {
            x: base[0] - outward[0] * tab_probe_y,
            y: base[1] - outward[1] * tab_probe_y,
        };

        // Tab: point is in modified but not in raw (added material)
        let in_mod_tab = mod_shape.contains(&tab_pt);
        let in_raw_tab = raw_shape.contains(&tab_pt);
        tabs[i] = in_mod_tab && !in_raw_tab;

        // Slot: point is in raw but not in modified (removed material)
        let in_mod_slot = mod_shape.contains(&slot_pt);
        let in_raw_slot = raw_shape.contains(&slot_pt);
        slots[i] = in_raw_slot && !in_mod_slot;
    }

    (tabs, slots)
}

fn evaluate_pair(
    add_a: &[bool],
    sub_a: &[bool],
    add_b: &[bool],
    sub_b: &[bool],
    max_shift: i32,
) -> (f64, f64, f64, i32) {
    let mut best_mismatch = f64::INFINITY;
    let mut best_collision: f64 = 0.0;
    let mut best_double_slot: f64 = 0.0;
    let mut best_shift: i32 = 0;

    for shift in -max_shift..=max_shift {
        let add_bs = shift_mask(add_b, shift);
        let sub_bs = shift_mask(sub_b, shift);

        let active: Vec<bool> = add_a
            .iter()
            .zip(sub_a.iter())
            .zip(add_bs.iter())
            .zip(sub_bs.iter())
            .map(|(((&a, &sa), &b), &sb)| a || sa || b || sb)
            .collect();

        // Classic mismatch
        let mismatch_classic = 0.5
            * (masked_xor_ratio(add_a, &sub_bs, &active)
                + masked_xor_ratio(&add_bs, sub_a, &active));

        // Preserve mode mismatch
        let not_sub_a: Vec<bool> = sub_a.iter().map(|&x| !x).collect();
        let not_sub_bs: Vec<bool> = sub_bs.iter().map(|&x| !x).collect();
        let mismatch_preserve = 0.5
            * (masked_xor_ratio(&not_sub_a, &sub_bs, &active)
                + masked_xor_ratio(&not_sub_bs, sub_a, &active));

        let add_cov = ratio_true(add_a) + ratio_true(&add_bs);
        let mismatch = if add_cov < 0.02 {
            mismatch_preserve
        } else {
            mismatch_classic.min(mismatch_preserve)
        };

        let collision = and_ratio(add_a, &add_bs);
        let double_slot = and_ratio(sub_a, &sub_bs);

        let key = (mismatch, collision, shift.unsigned_abs());
        let best_key = (best_mismatch, best_collision, best_shift.unsigned_abs());
        if key < best_key {
            best_mismatch = mismatch;
            best_collision = collision;
            best_double_slot = double_slot;
            best_shift = shift;
        }
    }

    (best_mismatch, best_collision, best_double_slot, best_shift)
}

struct JointEdge {
    joint_type: String,
    face_a_id: usize,
    face_b_id: usize,
    p1_a: Vec2,
    p2_a: Vec2,
    p1_b: Vec2,
    p2_b: Vec2,
}

fn collect_shared_joint_edges(
    projections: &HashMap<usize, Projection2D>,
    shared_edges: &[SharedEdge],
) -> Vec<JointEdge> {
    let mut joints = Vec::new();
    for se in shared_edges {
        let (a, b) = (se.face_a_id, se.face_b_id);
        if !projections.contains_key(&a) || !projections.contains_key(&b) {
            continue;
        }
        let proj_a = &projections[&a];
        let proj_b = &projections[&b];
        let idx_a = match find_matching_edge_index(proj_a, se) {
            Some(i) => i,
            None => continue,
        };
        let idx_b = match find_matching_edge_index(proj_b, se) {
            Some(i) => i,
            None => continue,
        };
        joints.push(JointEdge {
            joint_type: "shared".to_string(),
            face_a_id: a,
            face_b_id: b,
            p1_a: proj_a.outer_edges_2d[idx_a].0,
            p2_a: proj_a.outer_edges_2d[idx_a].1,
            p1_b: proj_b.outer_edges_2d[idx_b].0,
            p2_b: proj_b.outer_edges_2d[idx_b].1,
        });
    }
    joints
}

fn collect_through_slot_edges(
    projections: &HashMap<usize, Projection2D>,
    shared_edges: &[SharedEdge],
    bottom_id: usize,
    faces: Option<&[PlanarFace]>,
) -> Vec<JointEdge> {
    let faces = match faces {
        Some(f) => f,
        None => return Vec::new(),
    };
    let bottom_proj = match projections.get(&bottom_id) {
        Some(p) => p,
        None => return Vec::new(),
    };
    let face_map: HashMap<usize, &PlanarFace> = faces.iter().map(|f| (f.face_id, f)).collect();
    let bottom_face = match face_map.get(&bottom_id) {
        Some(f) => f,
        None => return Vec::new(),
    };

    let mut bottom_adjacent = std::collections::HashSet::new();
    for se in shared_edges {
        if se.face_a_id == bottom_id {
            bottom_adjacent.insert(se.face_b_id);
        } else if se.face_b_id == bottom_id {
            bottom_adjacent.insert(se.face_a_id);
        }
    }

    let mut joints = Vec::new();
    for (&wall_id, wall_proj) in projections {
        if wall_id == bottom_id || bottom_adjacent.contains(&wall_id) {
            continue;
        }
        let wall_face = match face_map.get(&wall_id) {
            Some(f) => f,
            None => continue,
        };
        let endpoints = match find_bottom_edge_endpoints(wall_face, bottom_face, 5.0) {
            Some(ep) => ep,
            None => continue,
        };
        let (p_start_3d, p_end_3d) = endpoints;

        let p1_bottom = project_point_3d_to_2d(
            p_start_3d,
            bottom_proj.origin_3d,
            bottom_proj.u_axis,
            bottom_proj.v_axis,
        );
        let p2_bottom = project_point_3d_to_2d(
            p_end_3d,
            bottom_proj.origin_3d,
            bottom_proj.u_axis,
            bottom_proj.v_axis,
        );
        let p1_wall = project_point_3d_to_2d(
            p_start_3d,
            wall_proj.origin_3d,
            wall_proj.u_axis,
            wall_proj.v_axis,
        );
        let p2_wall = project_point_3d_to_2d(
            p_end_3d,
            wall_proj.origin_3d,
            wall_proj.u_axis,
            wall_proj.v_axis,
        );

        joints.push(JointEdge {
            joint_type: "through_slot".to_string(),
            face_a_id: bottom_id,
            face_b_id: wall_id,
            p1_a: p1_bottom,
            p2_a: p2_bottom,
            p1_b: p1_wall,
            p2_b: p2_wall,
        });
    }

    joints
}

pub fn verify_joint_mesh(
    projections: &HashMap<usize, Projection2D>,
    modified_polygons: &HashMap<usize, Vec<Vec2>>,
    slot_cutouts: &HashMap<usize, Vec<Vec<Vec2>>>,
    shared_edges: &[SharedEdge],
    bottom_id: usize,
    thickness: f64,
    sample_step: f64,
    mismatch_tolerance: f64,
    run_interference: bool,
    interference_tolerance: f64,
    faces: Option<&[PlanarFace]>,
) -> VerificationReport {
    let raw_shapes: HashMap<usize, GeoPolygon<f64>> = projections
        .iter()
        .map(|(&fid, proj)| (fid, polygon_to_geo(&proj.outer_polygon, &proj.inner_polygons)))
        .collect();

    let mod_shapes: HashMap<usize, GeoPolygon<f64>> = projections
        .keys()
        .map(|&fid| {
            let poly = modified_polygons.get(&fid).cloned().unwrap_or_default();
            let holes = slot_cutouts.get(&fid).cloned().unwrap_or_default();
            (fid, polygon_to_geo(&poly, &holes))
        })
        .collect();

    let mut joints = collect_shared_joint_edges(projections, shared_edges);
    joints.extend(collect_through_slot_edges(
        projections,
        shared_edges,
        bottom_id,
        faces,
    ));

    let mut results = Vec::new();
    for (idx, joint) in joints.iter().enumerate() {
        let a = joint.face_a_id;
        let b = joint.face_b_id;

        let edge_len_a = dist2(joint.p1_a, joint.p2_a);
        let edge_len_b = dist2(joint.p1_b, joint.p2_b);
        let max_len = edge_len_a.max(edge_len_b);
        let samples = (max_len / sample_step.max(0.05)).ceil().max(80.0) as usize;
        let max_shift = ((0.5 / sample_step.max(0.05)).round() as i32).max(1);

        let outward_a = outward_direction_for_verify(joint.p1_a, joint.p2_a, &raw_shapes[&a]);
        let outward_b = outward_direction_for_verify(joint.p1_b, joint.p2_b, &raw_shapes[&b]);

        let (add_a, sub_a) = edge_feature_masks(
            &raw_shapes[&a],
            &mod_shapes[&a],
            joint.p1_a,
            joint.p2_a,
            outward_a,
            thickness,
            samples,
        );
        let (add_b_base, sub_b_base) = edge_feature_masks(
            &raw_shapes[&b],
            &mod_shapes[&b],
            joint.p1_b,
            joint.p2_b,
            outward_b,
            thickness,
            samples,
        );

        // Try both directions
        let mut best_mismatch = f64::INFINITY;
        let mut best_collision: f64 = 0.0;
        let mut best_double_slot: f64 = 0.0;
        let mut best_shift: i32 = 0;
        let mut best_reversed = false;
        let mut best_add_b = add_b_base.clone();
        let mut best_sub_b = sub_b_base.clone();

        for reversed in [false, true] {
            let add_b = if reversed {
                reverse_mask(&add_b_base)
            } else {
                add_b_base.clone()
            };
            let sub_b = if reversed {
                reverse_mask(&sub_b_base)
            } else {
                sub_b_base.clone()
            };

            let (mm, cr, dsr, shift) = evaluate_pair(&add_a, &sub_a, &add_b, &sub_b, max_shift);
            let key = (mm, cr, shift.unsigned_abs());
            let best_key = (best_mismatch, best_collision, best_shift.unsigned_abs());
            if key < best_key {
                best_mismatch = mm;
                best_collision = cr;
                best_double_slot = dsr;
                best_shift = shift;
                best_reversed = reversed;
                best_add_b = add_b;
                best_sub_b = sub_b;
            }
        }

        let has_features =
            [ratio_true(&add_a), ratio_true(&best_add_b), ratio_true(&sub_a), ratio_true(&best_sub_b)]
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max)
                > 0.01;

        let mut fail_reasons = Vec::new();
        if has_features && best_mismatch > mismatch_tolerance {
            fail_reasons.push(format!(
                "mismatch {:.3} > {:.3}",
                best_mismatch, mismatch_tolerance
            ));
        }
        if run_interference && best_collision > interference_tolerance {
            fail_reasons.push(format!(
                "collision {:.3} > {:.3}",
                best_collision, interference_tolerance
            ));
        }

        let passed = fail_reasons.is_empty();
        let reason = if passed {
            "ok".to_string()
        } else {
            fail_reasons.join("; ")
        };

        results.push(JointVerification {
            joint_id: format!("{}_{}_{}", joint.joint_type, a, b),
            joint_type: joint.joint_type.clone(),
            face_a_id: a,
            face_b_id: b,
            samples,
            reversed_b: best_reversed,
            shift_samples: best_shift,
            mismatch_ratio: best_mismatch,
            collision_ratio: best_collision,
            double_slot_ratio: best_double_slot,
            add_coverage_a: ratio_true(&add_a),
            add_coverage_b: ratio_true(&best_add_b),
            sub_coverage_a: ratio_true(&sub_a),
            sub_coverage_b: ratio_true(&best_sub_b),
            passed,
            reason,
            debug_svg: None,
        });
    }

    let failed = results.iter().filter(|r| !r.passed).count();
    VerificationReport {
        total_joints: results.len(),
        failed_joints: failed,
        run_interference,
        mismatch_tolerance,
        interference_tolerance,
        joints: results,
    }
}

pub fn write_verification_report(path: &str, report: &VerificationReport) -> anyhow::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)?;
    Ok(())
}
