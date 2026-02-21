use std::collections::{HashMap, HashSet, VecDeque};

use crate::math_utils::{dist3, dot3, norm3, normalize3, sub3};
use crate::types::{EdgeData, FaceClassification, PlanarFace, SharedEdge};

fn edge_length(edge: &EdgeData) -> f64 {
    dist3(edge.start, edge.end)
}

fn points_close(p1: [f64; 3], p2: [f64; 3], tol: f64) -> bool {
    (p1[0] - p2[0]).abs() < tol && (p1[1] - p2[1]).abs() < tol && (p1[2] - p2[2]).abs() < tol
}

fn edges_coincident(e1: &EdgeData, e2: &EdgeData, tol: f64) -> bool {
    // Forward match
    if points_close(e1.start, e2.start, tol) && points_close(e1.end, e2.end, tol) {
        return true;
    }
    // Reverse match
    if points_close(e1.start, e2.end, tol) && points_close(e1.end, e2.start, tol) {
        return true;
    }
    // Midpoint + length match
    if points_close(e1.midpoint, e2.midpoint, tol) {
        let len1 = edge_length(e1);
        let len2 = edge_length(e2);
        if (len1 - len2).abs() < tol {
            return true;
        }
    }
    false
}

/// Find edges shared between pairs of faces.
pub fn find_shared_edges(faces: &[PlanarFace], tol: f64) -> Vec<SharedEdge> {
    let mut shared = Vec::new();

    for i in 0..faces.len() {
        for j in (i + 1)..faces.len() {
            for ea in &faces[i].outer_wire_edges {
                for eb in &faces[j].outer_wire_edges {
                    if edges_coincident(ea, eb, tol) {
                        let length = edge_length(ea);
                        if length > tol {
                            shared.push(SharedEdge {
                                face_a_id: faces[i].face_id,
                                face_b_id: faces[j].face_id,
                                edge_a: ea.clone(),
                                edge_b: eb.clone(),
                                midpoint: ea.midpoint,
                                length,
                            });
                        }
                    }
                }
            }
        }
    }

    shared
}

/// Build adjacency graph: face_id -> [(neighbor_id, shared_edge), ...].
pub fn build_adjacency(
    faces: &[PlanarFace],
    shared_edges: &[SharedEdge],
) -> HashMap<usize, Vec<(usize, SharedEdge)>> {
    let mut adj: HashMap<usize, Vec<(usize, SharedEdge)>> = HashMap::new();
    for f in faces {
        adj.entry(f.face_id).or_default();
    }
    for se in shared_edges {
        adj.entry(se.face_a_id)
            .or_default()
            .push((se.face_b_id, se.clone()));
        adj.entry(se.face_b_id)
            .or_default()
            .push((se.face_a_id, se.clone()));
    }
    adj
}

/// Find pairs of faces with opposite normals and similar area (inner/outer of same panel).
fn find_opposite_pairs(
    faces: &[PlanarFace],
    tol: f64,
    max_offset: f64,
    max_lateral_offset: f64,
) -> HashMap<usize, usize> {
    let mut pairs = HashMap::new();
    let mut used = HashSet::new();

    let mut sorted_faces: Vec<&PlanarFace> = faces.iter().collect();
    sorted_faces.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap());

    for (i, fa) in sorted_faces.iter().enumerate() {
        if used.contains(&fa.face_id) {
            continue;
        }

        let n_len = norm3(fa.normal);
        if n_len < 1e-12 {
            continue;
        }
        let n_hat = normalize3(fa.normal);

        let mut best_candidate: Option<(f64, usize)> = None;

        for (j, fb) in sorted_faces.iter().enumerate() {
            if j <= i || used.contains(&fb.face_id) {
                continue;
            }

            let dot = dot3(fa.normal, fb.normal);
            if dot < -(1.0 - tol) {
                let area_ratio = fa.area.min(fb.area) / fa.area.max(fb.area);
                if area_ratio > 0.8 {
                    let delta = sub3(fb.center, fa.center);
                    let normal_proj = dot3(delta, n_hat);
                    let normal_offset = normal_proj.abs();
                    let lateral = [
                        delta[0] - normal_proj * n_hat[0],
                        delta[1] - normal_proj * n_hat[1],
                        delta[2] - normal_proj * n_hat[2],
                    ];
                    let lateral_offset = norm3(lateral);

                    if normal_offset <= max_offset && lateral_offset <= max_lateral_offset {
                        let score = normal_offset + 0.1 * lateral_offset;
                        if best_candidate.is_none() || score < best_candidate.unwrap().0 {
                            best_candidate = Some((score, fb.face_id));
                        }
                    }
                }
            }
        }

        if let Some((_, match_id)) = best_candidate {
            pairs.insert(fa.face_id, match_id);
            used.insert(fa.face_id);
            used.insert(match_id);
        }
    }

    pairs
}

/// Classify faces into bottom plate and walls.
pub fn classify_faces(
    faces: &[PlanarFace],
    shared_edges: &[SharedEdge],
) -> FaceClassification {
    if faces.is_empty() {
        panic!("No faces to classify");
    }

    let adjacency = build_adjacency(faces, shared_edges);
    let face_map: HashMap<usize, &PlanarFace> = faces.iter().map(|f| (f.face_id, f)).collect();

    // Find inner/outer pairs
    let pairs = find_opposite_pairs(faces, 0.05, 20.0, 20.0);
    let outer_ids: HashSet<usize> = pairs.keys().copied().collect();
    let inner_ids: HashSet<usize> = pairs.values().copied().collect();

    // Structural faces: outer faces from pairs + unpaired large faces
    let mut structural_ids: HashSet<usize> = outer_ids.clone();
    let max_area = faces.iter().map(|f| f.area).fold(0.0_f64, f64::max);

    for f in faces {
        if !outer_ids.contains(&f.face_id) && !inner_ids.contains(&f.face_id) {
            if f.area > max_area * 0.05 {
                structural_ids.insert(f.face_id);
            }
        }
    }

    if structural_ids.is_empty() {
        structural_ids = faces.iter().map(|f| f.face_id).collect();
    }

    // Bottom: largest structural face
    let bottom_id = structural_ids
        .iter()
        .max_by(|a, b| {
            face_map[a]
                .area
                .partial_cmp(&face_map[b].area)
                .unwrap()
        })
        .copied()
        .unwrap();

    // Walls: flood-fill from bottom through adjacency.
    // Traverse ALL faces (including inner faces) but only collect structural faces as walls.
    // This ensures we reach structural faces that are only adjacent to inner faces
    // (e.g. back wall connected through inner surfaces of side walls).
    let mut wall_ids = HashSet::new();
    let mut visited = HashSet::new();
    visited.insert(bottom_id);
    let mut queue = VecDeque::new();
    queue.push_back(bottom_id);

    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&current) {
            for (neighbor_id, _) in neighbors {
                if visited.contains(neighbor_id) {
                    continue;
                }
                visited.insert(*neighbor_id);
                queue.push_back(*neighbor_id);
                if structural_ids.contains(neighbor_id) {
                    wall_ids.insert(*neighbor_id);
                }
            }
        }
    }

    let bottom = face_map[&bottom_id].clone();
    let walls: Vec<PlanarFace> = wall_ids.iter().map(|id| face_map[id].clone()).collect();
    let classified_ids: HashSet<usize> = std::iter::once(bottom_id).chain(wall_ids).collect();
    let other: Vec<PlanarFace> = faces
        .iter()
        .filter(|f| !classified_ids.contains(&f.face_id))
        .cloned()
        .collect();

    FaceClassification {
        bottom,
        walls,
        other,
        adjacency,
    }
}
