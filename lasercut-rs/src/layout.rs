use std::collections::HashMap;

use crate::finger_joints::find_matching_edge_index;
use crate::math_utils::*;
use crate::projector::project_face;
use crate::types::*;

pub const DEFAULT_FOLDED_OFFSET: f64 = 10.0;

#[derive(Debug, Clone)]
pub struct ArrangedPart {
    pub fid: usize,
    pub label: String,
    pub polygon: Vec<Vec2>,
    pub holes: Vec<Vec<Vec2>>,
}

#[derive(Debug, Clone)]
pub struct PreparedPart {
    pub fid: usize,
    pub label: String,
    pub polygon: Vec<Vec2>,
    pub holes: Vec<Vec<Vec2>>,
    pub width: f64,
    pub height: f64,
    pub min_x: f64,
    pub min_y: f64,
}

pub fn prepare_parts(
    projections: &HashMap<usize, Projection2D>,
    modified_polygons: &HashMap<usize, Vec<Vec2>>,
    slot_cutouts: &HashMap<usize, Vec<Vec<Vec2>>>,
) -> Vec<PreparedPart> {
    let mut fids: Vec<usize> = modified_polygons.keys().copied().collect();
    fids.sort();

    let mut parts = Vec::new();
    for fid in fids {
        let polygon = &modified_polygons[&fid];
        let proj = &projections[&fid];
        if polygon.is_empty() {
            continue;
        }

        let cutouts = slot_cutouts.get(&fid).cloned().unwrap_or_default();
        let all_polys: Vec<&Vec<Vec2>> = std::iter::once(polygon).chain(cutouts.iter()).collect();

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for poly in &all_polys {
            for p in poly.iter() {
                xs.push(p[0]);
                ys.push(p[1]);
            }
        }

        let min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_y = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let norm_poly: Vec<Vec2> = polygon.iter().map(|p| [p[0] - min_x, p[1] - min_y]).collect();
        let norm_holes: Vec<Vec<Vec2>> = cutouts
            .iter()
            .map(|hole| hole.iter().map(|p| [p[0] - min_x, p[1] - min_y]).collect())
            .collect();

        parts.push(PreparedPart {
            fid,
            label: proj.label.clone(),
            polygon: norm_poly,
            holes: norm_holes,
            width: max_x - min_x,
            height: max_y - min_y,
            min_x,
            min_y,
        });
    }

    parts
}

// ── Packed layout ───────────────────────────────────────────────────────────

fn pack_parts(
    parts: &[PreparedPart],
    padding: f64,
    max_width: Option<f64>,
) -> (Vec<Option<(f64, f64, bool)>>, f64, f64) {
    let mut indexed: Vec<(usize, &PreparedPart)> = parts.iter().enumerate().collect();
    indexed.sort_by(|a, b| {
        let ma = a.1.width.max(a.1.height);
        let mb = b.1.width.max(b.1.height);
        mb.partial_cmp(&ma).unwrap()
    });

    let mut placed: Vec<(f64, f64, f64, f64, usize, bool)> = Vec::new(); // x,y,w,h,idx,rotated

    for (orig_idx, part) in &indexed {
        let (w0, h0) = (part.width, part.height);
        let mut best_pos: Option<(f64, f64, f64, f64)> = None;
        let mut best_y_max = f64::INFINITY;
        let mut best_rotated = false;

        for (rotated, w, h) in [(false, w0, h0), (true, h0, w0)] {
            let mut x_candidates: Vec<f64> = vec![0.0];
            for &(px, _py, pw, _ph, _, _) in &placed {
                x_candidates.push(px + pw + padding);
            }

            for x in &x_candidates {
                if let Some(mw) = max_width {
                    if x + w + padding > mw {
                        continue;
                    }
                }
                let mut y: f64 = 0.0;
                for &(px, py, pw, ph, _, _) in &placed {
                    if *x < px + pw + padding && x + w + padding > px {
                        y = y.max(py + ph + padding);
                    }
                }
                let y_max = y + h;
                if y_max < best_y_max {
                    best_y_max = y_max;
                    best_pos = Some((*x, y, w, h));
                    best_rotated = rotated;
                }
            }
        }

        if let Some((x, y, w, h)) = best_pos {
            placed.push((x, y, w, h, *orig_idx, best_rotated));
        }
    }

    let mut positions: Vec<Option<(f64, f64, bool)>> = vec![None; parts.len()];
    for &(x, y, _w, _h, orig_idx, rotated) in &placed {
        positions[orig_idx] = Some((x + padding, y + padding, rotated));
    }

    let total_w = placed
        .iter()
        .map(|(x, _y, w, _h, _, _)| x + w)
        .fold(0.0_f64, f64::max)
        + 2.0 * padding;
    let total_h = placed
        .iter()
        .map(|(_x, y, _w, h, _, _)| y + h)
        .fold(0.0_f64, f64::max)
        + 2.0 * padding;

    (positions, total_w, total_h)
}

fn rotate_polygon_90(polygon: &[Vec2], w: f64) -> Vec<Vec2> {
    polygon.iter().map(|p| [p[1], w - p[0]]).collect()
}

fn transform_part_packed(
    part: &PreparedPart,
    ox: f64,
    oy: f64,
    rotated: bool,
) -> (Vec<Vec2>, Vec<Vec<Vec2>>) {
    let mut poly = part.polygon.clone();
    let mut holes = part.holes.clone();

    if rotated {
        poly = rotate_polygon_90(&poly, part.width);
        holes = holes
            .iter()
            .map(|h| rotate_polygon_90(h, part.width))
            .collect();
    }

    poly = poly.iter().map(|p| [p[0] + ox, p[1] + oy]).collect();
    holes = holes
        .iter()
        .map(|h| h.iter().map(|p| [p[0] + ox, p[1] + oy]).collect())
        .collect();

    (poly, holes)
}

pub fn arrange_packed(
    parts: &[PreparedPart],
    padding: f64,
    max_width: Option<f64>,
) -> (Vec<ArrangedPart>, f64, f64) {
    let (positions, total_w, total_h) = pack_parts(parts, padding, max_width);
    let mut geoms = Vec::new();

    for (part, pos) in parts.iter().zip(positions.iter()) {
        if let Some((ox, oy, rotated)) = pos {
            let (poly, holes) = transform_part_packed(part, *ox, *oy, *rotated);
            geoms.push(ArrangedPart {
                fid: part.fid,
                label: part.label.clone(),
                polygon: poly,
                holes,
            });
        }
    }

    (geoms, total_w, total_h)
}

// ── Folded/unfolded layout ──────────────────────────────────────────────────

fn transform_points_to_edge(
    points: &[Vec2],
    src_a: Vec2,
    src_b: Vec2,
    dst_a: Vec2,
    dst_b: Vec2,
) -> Vec<Vec2> {
    let v_src = sub2(src_b, src_a);
    let v_dst = sub2(dst_b, dst_a);

    let src_len = norm2(v_src);
    let dst_len = norm2(v_dst);

    if src_len < 1e-9 || dst_len < 1e-9 {
        let tx = dst_a[0] - src_a[0];
        let ty = dst_a[1] - src_a[1];
        return points.iter().map(|p| [p[0] + tx, p[1] + ty]).collect();
    }

    let ang_src = v_src[1].atan2(v_src[0]);
    let ang_dst = v_dst[1].atan2(v_dst[0]);
    let ang = ang_dst - ang_src;
    let c = ang.cos();
    let s = ang.sin();

    points
        .iter()
        .map(|p| {
            let dx = p[0] - src_a[0];
            let dy = p[1] - src_a[1];
            [c * dx - s * dy + dst_a[0], s * dx + c * dy + dst_a[1]]
        })
        .collect()
}

fn reflect_points_across_line(points: &[Vec2], a: Vec2, b: Vec2) -> Vec<Vec2> {
    let vx = b[0] - a[0];
    let vy = b[1] - a[1];
    let ln = (vx * vx + vy * vy).sqrt();
    if ln < 1e-12 {
        return points.to_vec();
    }
    let ux = vx / ln;
    let uy = vy / ln;

    points
        .iter()
        .map(|p| {
            let rx = p[0] - a[0];
            let ry = p[1] - a[1];
            let dot = rx * ux + ry * uy;
            let px = dot * ux;
            let py = dot * uy;
            let fx = 2.0 * px - rx;
            let fy = 2.0 * py - ry;
            [a[0] + fx, a[1] + fy]
        })
        .collect()
}

fn centroid(points: &[Vec2]) -> Vec2 {
    if points.is_empty() {
        return [0.0, 0.0];
    }
    let sum: Vec2 = points.iter().fold([0.0, 0.0], |acc, p| [acc[0] + p[0], acc[1] + p[1]]);
    let n = points.len() as f64;
    [sum[0] / n, sum[1] / n]
}

fn outward_direction_simple(p1: Vec2, p2: Vec2, polygon: &[Vec2]) -> Vec2 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let ln = (dx * dx + dy * dy).sqrt();
    if ln < 1e-12 {
        return [0.0, -1.0];
    }

    let n1 = [-dy / ln, dx / ln];
    let n2 = [dy / ln, -dx / ln];

    let c = centroid(polygon);
    let mx = (p1[0] + p2[0]) * 0.5;
    let my = (p1[1] + p2[1]) * 0.5;

    let d1 = (mx + n1[0] - c[0]).powi(2) + (my + n1[1] - c[1]).powi(2);
    let d2 = (mx + n2[0] - c[0]).powi(2) + (my + n2[1] - c[1]).powi(2);

    if d1 > d2 {
        n1
    } else {
        n2
    }
}

fn edges_reversed_3d(
    proj_a: &Projection2D,
    idx_a: usize,
    proj_b: &Projection2D,
    idx_b: usize,
    tol: f64,
) -> bool {
    let ea = &proj_a.edge_map_3d[idx_a];
    let eb = &proj_b.edge_map_3d[idx_b];
    let fwd = points_close3(ea.start, eb.start, tol) && points_close3(ea.end, eb.end, tol);
    let rev = points_close3(ea.start, eb.end, tol) && points_close3(ea.end, eb.start, tol);
    rev && !fwd
}

struct BottomAnchor {
    bottom_edge: (Vec2, Vec2),
    wall_edge: (Vec2, Vec2),
    reversed: bool,
    through_slot: bool,
}

fn build_bottom_anchors(
    projections: &HashMap<usize, Projection2D>,
    shared_edges: &[SharedEdge],
    bottom_id: usize,
    faces: Option<&[PlanarFace]>,
) -> HashMap<usize, BottomAnchor> {
    let mut anchors = HashMap::new();
    let bottom_proj = match projections.get(&bottom_id) {
        Some(p) => p,
        None => return anchors,
    };

    // Direct bottom shared edges
    for se in shared_edges {
        if se.face_a_id != bottom_id && se.face_b_id != bottom_id {
            continue;
        }
        let wall_id = if se.face_a_id == bottom_id {
            se.face_b_id
        } else {
            se.face_a_id
        };
        let wall_proj = match projections.get(&wall_id) {
            Some(p) => p,
            None => continue,
        };

        let bottom_idx = match find_matching_edge_index(bottom_proj, se) {
            Some(idx) => idx,
            None => continue,
        };
        let wall_idx = match find_matching_edge_index(wall_proj, se) {
            Some(idx) => idx,
            None => continue,
        };

        let bp = bottom_proj.outer_edges_2d[bottom_idx];
        let wp = wall_proj.outer_edges_2d[wall_idx];
        let reversed = edges_reversed_3d(bottom_proj, bottom_idx, wall_proj, wall_idx, 0.5);

        anchors.insert(
            wall_id,
            BottomAnchor {
                bottom_edge: bp,
                wall_edge: wp,
                reversed,
                through_slot: false,
            },
        );
    }

    // Through-slot walls
    if let Some(all_faces) = faces {
        use crate::finger_joints::find_bottom_edge_endpoints;

        let face_map: HashMap<usize, &PlanarFace> =
            all_faces.iter().map(|f| (f.face_id, f)).collect();
        let bottom_face = match face_map.get(&bottom_id) {
            Some(f) => f,
            None => return anchors,
        };

        let mut bottom_adjacent = std::collections::HashSet::new();
        for se in shared_edges {
            if se.face_a_id == bottom_id {
                bottom_adjacent.insert(se.face_b_id);
            } else if se.face_b_id == bottom_id {
                bottom_adjacent.insert(se.face_a_id);
            }
        }

        for (&fid, wall_proj) in projections {
            if fid == bottom_id || anchors.contains_key(&fid) || bottom_adjacent.contains(&fid) {
                continue;
            }
            let wall_face = match face_map.get(&fid) {
                Some(f) => f,
                None => continue,
            };

            let endpoints = match find_bottom_edge_endpoints(wall_face, bottom_face, 5.0) {
                Some(ep) => ep,
                None => continue,
            };
            let (p_start_3d, p_end_3d) = endpoints;

            let bp1 = project_point_3d_to_2d(
                p_start_3d,
                bottom_proj.origin_3d,
                bottom_proj.u_axis,
                bottom_proj.v_axis,
            );
            let bp2 = project_point_3d_to_2d(
                p_end_3d,
                bottom_proj.origin_3d,
                bottom_proj.u_axis,
                bottom_proj.v_axis,
            );
            let wp1 = project_point_3d_to_2d(
                p_start_3d,
                wall_proj.origin_3d,
                wall_proj.u_axis,
                wall_proj.v_axis,
            );
            let wp2 = project_point_3d_to_2d(
                p_end_3d,
                wall_proj.origin_3d,
                wall_proj.u_axis,
                wall_proj.v_axis,
            );

            anchors.insert(
                fid,
                BottomAnchor {
                    bottom_edge: (bp1, bp2),
                    wall_edge: (wp1, wp2),
                    reversed: false,
                    through_slot: true,
                },
            );
        }
    }

    anchors
}

fn pt_to_local(part: &PreparedPart, pt: Vec2) -> Vec2 {
    [pt[0] - part.min_x, pt[1] - part.min_y]
}

fn shift_geoms_to_positive(
    geoms: Vec<ArrangedPart>,
    padding: f64,
) -> (Vec<ArrangedPart>, f64, f64) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for g in &geoms {
        for p in &g.polygon {
            xs.push(p[0]);
            ys.push(p[1]);
        }
        for hole in &g.holes {
            for p in hole {
                xs.push(p[0]);
                ys.push(p[1]);
            }
        }
    }

    if xs.is_empty() {
        return (geoms, 2.0 * padding, 2.0 * padding);
    }

    let min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let dx = padding - min_x;
    let dy = padding - min_y;

    let shifted = geoms
        .into_iter()
        .map(|g| ArrangedPart {
            fid: g.fid,
            label: g.label,
            polygon: g.polygon.iter().map(|p| [p[0] + dx, p[1] + dy]).collect(),
            holes: g
                .holes
                .iter()
                .map(|h| h.iter().map(|p| [p[0] + dx, p[1] + dy]).collect())
                .collect(),
        })
        .collect();

    let total_w = (max_x - min_x) + 2.0 * padding;
    let total_h = (max_y - min_y) + 2.0 * padding;
    (shifted, total_w, total_h)
}

pub fn arrange_folded(
    parts: &[PreparedPart],
    projections: &HashMap<usize, Projection2D>,
    shared_edges: &[SharedEdge],
    bottom_id: usize,
    faces: Option<&[PlanarFace]>,
    wall_offset: f64,
    padding: f64,
) -> (Vec<ArrangedPart>, f64, f64) {
    let part_by_fid: HashMap<usize, &PreparedPart> = parts.iter().map(|p| (p.fid, p)).collect();

    let bottom_part = match part_by_fid.get(&bottom_id) {
        Some(p) => p,
        None => return arrange_packed(parts, padding, None),
    };

    let anchors = build_bottom_anchors(projections, shared_edges, bottom_id, faces);

    let mut geoms_by_fid: HashMap<usize, ArrangedPart> = HashMap::new();
    geoms_by_fid.insert(
        bottom_id,
        ArrangedPart {
            fid: bottom_part.fid,
            label: bottom_part.label.clone(),
            polygon: bottom_part.polygon.clone(),
            holes: bottom_part.holes.clone(),
        },
    );

    // Build transform closures (here we'll store the transforms as edge pairs)
    struct PlacementTransform {
        src_a: Vec2,
        src_b: Vec2,
        dst_a: Vec2,
        dst_b: Vec2,
        reflected: bool,
    }
    let mut transforms: HashMap<usize, PlacementTransform> = HashMap::new();

    let place_from_edge = |source_id: usize,
                           target_id: usize,
                           src_edge: (Vec2, Vec2),
                           dst_edge: (Vec2, Vec2),
                           reversed_edge: bool,
                           offset: f64,
                           geoms: &mut HashMap<usize, ArrangedPart>,
                           tfms: &mut HashMap<usize, PlacementTransform>|
     -> bool {
        let source_part = match part_by_fid.get(&source_id) {
            Some(p) => p,
            None => return false,
        };
        let target_geom = match geoms.get(&target_id) {
            Some(g) => g,
            None => return false,
        };

        let sp1 = pt_to_local(source_part, src_edge.0);
        let sp2 = pt_to_local(source_part, src_edge.1);

        // Transform destination edge through target's placement
        let dp1_local = match part_by_fid.get(&target_id) {
            Some(tp) => pt_to_local(tp, dst_edge.0),
            None => return false,
        };
        let dp2_local = match part_by_fid.get(&target_id) {
            Some(tp) => pt_to_local(tp, dst_edge.1),
            None => return false,
        };

        // If target has a transform, apply it
        let (dp1, dp2) = if let Some(tfm) = tfms.get(&target_id) {
            let pts = transform_points_to_edge(
                &[dp1_local, dp2_local],
                tfm.src_a,
                tfm.src_b,
                tfm.dst_a,
                tfm.dst_b,
            );
            let pts = if tfm.reflected {
                reflect_points_across_line(&pts, tfm.dst_a, tfm.dst_b)
            } else {
                pts
            };
            (pts[0], pts[1])
        } else {
            (dp1_local, dp2_local)
        };

        let outward = outward_direction_simple(dp1, dp2, &target_geom.polygon);
        let mut tp1 = [dp1[0] + outward[0] * offset, dp1[1] + outward[1] * offset];
        let mut tp2 = [dp2[0] + outward[0] * offset, dp2[1] + outward[1] * offset];
        if reversed_edge {
            std::mem::swap(&mut tp1, &mut tp2);
        }

        let poly = transform_points_to_edge(&source_part.polygon, sp1, sp2, tp1, tp2);
        let holes: Vec<Vec<Vec2>> = source_part
            .holes
            .iter()
            .map(|h| transform_points_to_edge(h, sp1, sp2, tp1, tp2))
            .collect();

        // Check if placement is on the correct side
        let c = centroid(&poly);
        let m = [(tp1[0] + tp2[0]) * 0.5, (tp1[1] + tp2[1]) * 0.5];
        let side = (c[0] - m[0]) * outward[0] + (c[1] - m[1]) * outward[1];

        let (final_poly, final_holes, reflected) = if side < 0.0 {
            let rp = reflect_points_across_line(&poly, tp1, tp2);
            let rh: Vec<Vec<Vec2>> = holes
                .iter()
                .map(|h| reflect_points_across_line(h, tp1, tp2))
                .collect();
            (rp, rh, true)
        } else {
            (poly, holes, false)
        };

        geoms.insert(
            source_id,
            ArrangedPart {
                fid: source_part.fid,
                label: source_part.label.clone(),
                polygon: final_poly,
                holes: final_holes,
            },
        );
        tfms.insert(
            source_id,
            PlacementTransform {
                src_a: sp1,
                src_b: sp2,
                dst_a: tp1,
                dst_b: tp2,
                reflected,
            },
        );
        true
    };

    let mut placed = std::collections::HashSet::new();
    placed.insert(bottom_id);

    // Place bottom-anchored walls first
    for (&fid, anchor) in &anchors {
        if placed.contains(&fid) {
            continue;
        }
        let ok = place_from_edge(
            fid,
            bottom_id,
            anchor.wall_edge,
            anchor.bottom_edge,
            anchor.reversed,
            wall_offset,
            &mut geoms_by_fid,
            &mut transforms,
        );
        if ok {
            placed.insert(fid);
        }
    }

    // BFS expansion via shared edges
    let mut unplaced: std::collections::HashSet<usize> = parts
        .iter()
        .map(|p| p.fid)
        .filter(|fid| !placed.contains(fid))
        .collect();

    // Build shared relations
    let mut relations: HashMap<(usize, usize), (Vec2, Vec2, Vec2, Vec2, bool)> = HashMap::new();
    for se in shared_edges {
        let (a, b) = (se.face_a_id, se.face_b_id);
        if !projections.contains_key(&a) || !projections.contains_key(&b) {
            continue;
        }
        let pa = &projections[&a];
        let pb = &projections[&b];
        let ia = match find_matching_edge_index(pa, se) {
            Some(i) => i,
            None => continue,
        };
        let ib = match find_matching_edge_index(pb, se) {
            Some(i) => i,
            None => continue,
        };

        let (ap1, ap2) = pa.outer_edges_2d[ia];
        let (bp1, bp2) = pb.outer_edges_2d[ib];
        let rev_ab = edges_reversed_3d(pb, ib, pa, ia, 0.5);
        let rev_ba = edges_reversed_3d(pa, ia, pb, ib, 0.5);

        relations.insert((a, b), (ap1, ap2, bp1, bp2, rev_ab));
        relations.insert((b, a), (bp1, bp2, ap1, ap2, rev_ba));
    }

    loop {
        let mut progress = false;
        let fids: Vec<usize> = unplaced.iter().copied().collect();
        for fid in fids {
            let targets: Vec<usize> = placed
                .iter()
                .filter(|&&t| relations.contains_key(&(fid, t)))
                .copied()
                .collect();
            if targets.is_empty() {
                continue;
            }
            let target_id = targets
                .iter()
                .find(|&&t| t != bottom_id)
                .copied()
                .unwrap_or(targets[0]);

            let &(src_p1, src_p2, dst_p1, dst_p2, reversed) = &relations[&(fid, target_id)];
            let offset = if target_id == bottom_id {
                wall_offset
            } else {
                0.0
            };

            let ok = place_from_edge(
                fid,
                target_id,
                (src_p1, src_p2),
                (dst_p1, dst_p2),
                reversed,
                offset,
                &mut geoms_by_fid,
                &mut transforms,
            );
            if ok {
                placed.insert(fid);
                unplaced.remove(&fid);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }

    // Collect results
    let mut geoms: Vec<ArrangedPart> = Vec::new();
    let mut fids: Vec<usize> = geoms_by_fid.keys().copied().collect();
    fids.sort();
    for fid in fids {
        geoms.push(geoms_by_fid.remove(&fid).unwrap());
    }

    // Pack any leftover parts below
    let leftover_parts: Vec<PreparedPart> = parts
        .iter()
        .filter(|p| !placed.contains(&p.fid))
        .cloned()
        .collect();
    if !leftover_parts.is_empty() {
        let (packed, _, _) = arrange_packed(&leftover_parts, padding, None);
        let cur_max_y = geoms
            .iter()
            .flat_map(|g| g.polygon.iter().map(|p| p[1]))
            .fold(0.0_f64, f64::max);

        for g in packed {
            geoms.push(ArrangedPart {
                fid: g.fid,
                label: g.label,
                polygon: g
                    .polygon
                    .iter()
                    .map(|p| [p[0], p[1] + cur_max_y + wall_offset])
                    .collect(),
                holes: g
                    .holes
                    .iter()
                    .map(|h| {
                        h.iter()
                            .map(|p| [p[0], p[1] + cur_max_y + wall_offset])
                            .collect()
                    })
                    .collect(),
            });
        }
    }

    shift_geoms_to_positive(geoms, padding)
}

pub fn compute_arrangement(
    parts: &[PreparedPart],
    layout: &str,
    projections: &HashMap<usize, Projection2D>,
    shared_edges: &[SharedEdge],
    bottom_id: usize,
    faces: Option<&[PlanarFace]>,
    padding: f64,
    wall_offset: f64,
    pack_max_width: Option<f64>,
) -> (Vec<ArrangedPart>, f64, f64) {
    match layout {
        "unfolded" => arrange_folded(
            parts,
            projections,
            shared_edges,
            bottom_id,
            faces,
            wall_offset,
            padding,
        ),
        _ => arrange_packed(parts, padding, pack_max_width),
    }
}
