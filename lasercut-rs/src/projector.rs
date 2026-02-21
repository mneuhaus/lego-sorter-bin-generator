use crate::math_utils::*;
use crate::types::{EdgeData, PlanarFace, Projection2D, Vec2, Vec3};

/// Order edge endpoints to form a continuous polygon.
fn ordered_vertices_from_edges(edges: &[EdgeData], tol: f64) -> Vec<Vec3> {
    if edges.is_empty() {
        return Vec::new();
    }

    let mut remaining: Vec<usize> = (0..edges.len()).collect();
    let first = remaining.remove(0);
    let mut chain: Vec<&EdgeData> = vec![&edges[first]];

    while !remaining.is_empty() {
        let current_end = chain.last().unwrap().end;
        let mut best_idx = None;
        let mut best_dist = f64::INFINITY;
        let mut best_reversed = false;

        for &idx in &remaining {
            let e = &edges[idx];
            let d_start = dist3(e.start, current_end);
            let d_end = dist3(e.end, current_end);
            if d_start < best_dist && d_start <= tol {
                best_dist = d_start;
                best_idx = Some(idx);
                best_reversed = false;
            }
            if d_end < best_dist && d_end <= tol {
                best_dist = d_end;
                best_idx = Some(idx);
                best_reversed = true;
            }
        }

        if let Some(idx) = best_idx {
            remaining.retain(|&i| i != idx);
            if best_reversed {
                // We need to create a reversed edge - but we just store the direction info
                // For vertex extraction we just need the chain direction
                chain.push(&edges[idx]);
            } else {
                chain.push(&edges[idx]);
            }

            // Track actual vertex order
        } else {
            break;
        }
    }

    // Actually, let me redo this properly to handle reversed edges
    let mut remaining: Vec<usize> = (1..edges.len()).collect();
    let mut ordered_edges: Vec<EdgeData> = vec![edges[0].clone()];

    while !remaining.is_empty() {
        let current_end = ordered_edges.last().unwrap().end;
        let mut best_pos = None;
        let mut best_dist = f64::INFINITY;
        let mut best_reversed = false;

        for (pos, &idx) in remaining.iter().enumerate() {
            let e = &edges[idx];
            let d_start = dist3(e.start, current_end);
            let d_end = dist3(e.end, current_end);
            if d_start < best_dist && d_start <= tol {
                best_dist = d_start;
                best_pos = Some(pos);
                best_reversed = false;
            }
            if d_end < best_dist && d_end <= tol {
                best_dist = d_end;
                best_pos = Some(pos);
                best_reversed = true;
            }
        }

        if let Some(pos) = best_pos {
            let idx = remaining.remove(pos);
            let e = &edges[idx];
            if best_reversed {
                ordered_edges.push(EdgeData {
                    start: e.end,
                    end: e.start,
                    midpoint: e.midpoint,
                });
            } else {
                ordered_edges.push(e.clone());
            }
        } else {
            break;
        }
    }

    ordered_edges.iter().map(|e| e.start).collect()
}

/// Project a planar face to 2D.
pub fn project_face(face: &PlanarFace, label: &str) -> Projection2D {
    let normal = normalize3(face.normal);

    // Choose U axis from the first edge direction
    let u_raw = if !face.outer_wire_edges.is_empty() {
        let e0 = &face.outer_wire_edges[0];
        sub3(e0.end, e0.start)
    } else {
        // Fallback: pick an arbitrary direction perpendicular to normal
        if normal[2].abs() < 0.9 {
            cross3(normal, [0.0, 0.0, 1.0])
        } else {
            cross3(normal, [1.0, 0.0, 0.0])
        }
    };

    let u_axis = normalize3(u_raw);
    let v_axis = normalize3(cross3(normal, u_axis));

    // Origin = first vertex of outer wire or face center
    let origin = if !face.outer_wire_edges.is_empty() {
        face.outer_wire_edges[0].start
    } else {
        face.center
    };

    // Project outer wire vertices
    let outer_verts_3d = ordered_vertices_from_edges(&face.outer_wire_edges, 0.5);
    let outer_polygon: Vec<Vec2> = outer_verts_3d
        .iter()
        .map(|v| project_point_3d_to_2d(*v, origin, u_axis, v_axis))
        .collect();

    // Project outer edges (preserving edge correspondence)
    let mut outer_edges_2d = Vec::new();
    let mut edge_map_3d = Vec::new();
    for edge in &face.outer_wire_edges {
        let p1 = project_point_3d_to_2d(edge.start, origin, u_axis, v_axis);
        let p2 = project_point_3d_to_2d(edge.end, origin, u_axis, v_axis);
        outer_edges_2d.push((p1, p2));
        edge_map_3d.push(edge.clone());
    }

    // Project inner wires (holes)
    let inner_polygons: Vec<Vec<Vec2>> = face
        .inner_wires_edges
        .iter()
        .map(|inner_edges| {
            let verts = ordered_vertices_from_edges(inner_edges, 0.5);
            verts
                .iter()
                .map(|v| project_point_3d_to_2d(*v, origin, u_axis, v_axis))
                .collect()
        })
        .collect();

    Projection2D {
        face_id: face.face_id,
        label: label.to_string(),
        outer_polygon,
        inner_polygons,
        origin_3d: origin,
        u_axis,
        v_axis,
        normal,
        outer_edges_2d,
        edge_map_3d,
    }
}
