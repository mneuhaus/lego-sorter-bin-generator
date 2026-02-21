use std::collections::{HashMap, HashSet};

use geo::algorithm::area::Area;
use geo::algorithm::contains::Contains;
use geo::{Coord, LineString, Polygon as GeoPolygon};
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;

use crate::math_utils::*;
use crate::types::*;

pub const DEFAULT_FINGER_WIDTH: f64 = 20.0;
pub const DEFAULT_EDGE_MARGIN: f64 = 10.0;
pub const DEFAULT_NOTCH_BUFFER: f64 = 2.0;
pub const DEFAULT_PLATEAU_INSET: f64 = 3.0;
pub const DEFAULT_MIN_PLATEAU_LENGTH: f64 = 12.0;

// ── Polygon conversion helpers ──────────────────────────────────────────────

fn vec2_to_coord(p: Vec2) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

fn coord_to_vec2(c: Coord<f64>) -> Vec2 {
    [c.x, c.y]
}

fn polygon_to_geo(outer: &[Vec2], holes: &[Vec<Vec2>]) -> GeoPolygon<f64> {
    if outer.len() < 3 {
        return GeoPolygon::new(LineString::new(vec![]), vec![]);
    }
    let mut outer_coords: Vec<Coord<f64>> = outer.iter().map(|p| vec2_to_coord(*p)).collect();
    // Close the ring
    if outer_coords.first() != outer_coords.last() {
        outer_coords.push(outer_coords[0]);
    }
    let exterior = LineString::new(outer_coords);

    let interiors: Vec<LineString<f64>> = holes
        .iter()
        .filter(|h| h.len() >= 3)
        .map(|h| {
            let mut coords: Vec<Coord<f64>> = h.iter().map(|p| vec2_to_coord(*p)).collect();
            if coords.first() != coords.last() {
                coords.push(coords[0]);
            }
            LineString::new(coords)
        })
        .collect();

    GeoPolygon::new(exterior, interiors)
}

fn geo_to_vertices(poly: &GeoPolygon<f64>) -> (Vec<Vec2>, Vec<Vec<Vec2>>) {
    let outer: Vec<Vec2> = poly
        .exterior()
        .coords()
        .map(|c| coord_to_vec2(*c))
        .collect();
    // Remove closing duplicate
    let outer = simplify_ring(&outer, 0.15);

    let inners: Vec<Vec<Vec2>> = poly
        .interiors()
        .iter()
        .map(|ring| {
            let pts: Vec<Vec2> = ring.coords().map(|c| coord_to_vec2(*c)).collect();
            simplify_ring(&pts, 0.15)
        })
        .collect();

    (outer, inners)
}

fn simplify_ring(coords: &[Vec2], tol: f64) -> Vec<Vec2> {
    if coords.len() < 4 {
        return coords.to_vec();
    }
    let mut out = vec![coords[0]];
    for pt in &coords[1..] {
        if dist2(*out.last().unwrap(), *pt) >= tol {
            out.push(*pt);
        }
    }
    // Close check
    if out.len() > 1 && dist2(*out.last().unwrap(), out[0]) < tol {
        out.pop();
    }
    if out.len() >= 3 {
        out
    } else {
        coords.to_vec()
    }
}

// ── Boolean operations using i_overlay ──────────────────────────────────────

fn geo_polygon_to_paths(poly: &GeoPolygon<f64>) -> Vec<Vec<[f64; 2]>> {
    let mut paths = Vec::new();

    // i_overlay expects OPEN contours (no closing duplicate point).
    // geo's coords() returns N+1 points where first == last; we must strip it.
    let mut ext: Vec<[f64; 2]> = poly.exterior().coords().map(|c| [c.x, c.y]).collect();
    if ext.len() > 1 && ext.first() == ext.last() {
        ext.pop();
    }
    if ext.len() >= 3 {
        paths.push(ext);
    }

    for interior in poly.interiors() {
        let mut inner: Vec<[f64; 2]> = interior.coords().map(|c| [c.x, c.y]).collect();
        if inner.len() > 1 && inner.first() == inner.last() {
            inner.pop();
        }
        if inner.len() >= 3 {
            paths.push(inner);
        }
    }

    paths
}

fn paths_to_geo_polygon(paths: &[Vec<[f64; 2]>]) -> GeoPolygon<f64> {
    if paths.is_empty() {
        return GeoPolygon::new(LineString::new(vec![]), vec![]);
    }

    let exterior = LineString::new(
        paths[0].iter().map(|p| Coord { x: p[0], y: p[1] }).collect(),
    );

    let interiors: Vec<LineString<f64>> = paths[1..]
        .iter()
        .map(|path| {
            LineString::new(path.iter().map(|p| Coord { x: p[0], y: p[1] }).collect())
        })
        .collect();

    GeoPolygon::new(exterior, interiors)
}

fn overlay_op(
    subject: &GeoPolygon<f64>,
    clip: &GeoPolygon<f64>,
    rule: OverlayRule,
) -> GeoPolygon<f64> {
    let subj_paths = geo_polygon_to_paths(subject);
    let clip_paths = geo_polygon_to_paths(clip);

    if subj_paths.is_empty() || clip_paths.is_empty() {
        return subject.clone();
    }

    let result: Vec<Vec<Vec<[f64; 2]>>> =
        subj_paths.overlay(&clip_paths, rule, FillRule::EvenOdd);

    if result.is_empty() {
        return subject.clone();
    }

    // Reconstruct: each result element is a shape (exterior + holes).
    // Find the largest exterior and collect all holes.
    let mut best_ext_idx = 0;
    let mut best_ext_area = 0.0_f64;
    let mut all_holes: Vec<LineString<f64>> = Vec::new();

    for (i, shape) in result.iter().enumerate() {
        if shape.is_empty() {
            continue;
        }
        // First contour in shape is the exterior ring
        let ext_coords: Vec<Coord<f64>> =
            shape[0].iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
        let ext_ring = LineString::new(ext_coords);
        let area = GeoPolygon::new(ext_ring.clone(), vec![]).unsigned_area();
        if area > best_ext_area {
            best_ext_area = area;
            best_ext_idx = i;
        }
        // Subsequent contours are holes
        for hole_path in &shape[1..] {
            let hole_coords: Vec<Coord<f64>> =
                hole_path.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
            all_holes.push(LineString::new(hole_coords));
        }
    }

    let best_shape = &result[best_ext_idx];
    if best_shape.is_empty() {
        return subject.clone();
    }

    let ext_coords: Vec<Coord<f64>> =
        best_shape[0].iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
    let exterior = LineString::new(ext_coords);

    GeoPolygon::new(exterior, all_holes)
}

fn boolean_union(subject: &GeoPolygon<f64>, clip: &GeoPolygon<f64>) -> GeoPolygon<f64> {
    overlay_op(subject, clip, OverlayRule::Union)
}

fn boolean_difference(subject: &GeoPolygon<f64>, clip: &GeoPolygon<f64>) -> GeoPolygon<f64> {
    overlay_op(subject, clip, OverlayRule::Difference)
}

fn boolean_union_multi(subject: &GeoPolygon<f64>, clips: &[GeoPolygon<f64>]) -> GeoPolygon<f64> {
    let mut result = subject.clone();
    for clip in clips {
        result = boolean_union(&result, clip);
    }
    result
}

fn boolean_difference_multi(subject: &GeoPolygon<f64>, clips: &[GeoPolygon<f64>]) -> GeoPolygon<f64> {
    let mut result = subject.clone();
    for clip in clips {
        result = boolean_difference(&result, clip);
    }
    result
}

// ── Buffer / offset ─────────────────────────────────────────────────────────

fn buffer_polygon(poly: &GeoPolygon<f64>, distance: f64) -> GeoPolygon<f64> {
    if distance.abs() < 1e-9 {
        return poly.clone();
    }
    // Simple polygon offset using i_overlay is not directly available.
    // For kerf compensation, we use a simple approach: offset each edge outward.
    // This is a simplified implementation - for production, use a proper offset library.
    // For now, we'll implement a basic Minkowski-style buffer.
    //
    // Actually, let's use a simple approach: create a small rectangle at each edge
    // and union everything. This is the "dilation" approach.
    // For small offsets (kerf/2 is typically 0.02-0.1mm), this works fine.
    poly.clone() // TODO: implement proper polygon offset
}

// ── Plateau detection ───────────────────────────────────────────────────────

fn find_plateau_segments(
    p1: Vec2,
    p2: Vec2,
    polygon: &GeoPolygon<f64>,
    sample_interval: f64,
    plateau_inset: f64,
) -> Vec<(f64, f64)> {
    let edge_len = dist2(p1, p2);
    if edge_len < 1e-6 {
        return Vec::new();
    }

    let n_samples = (edge_len / sample_interval).ceil().max(20.0) as usize;
    let boundary = polygon.exterior();

    // Sample along edge, measuring distance to polygon boundary
    let mut distances = Vec::with_capacity(n_samples + 1);
    for i in 0..=n_samples {
        let t = i as f64 / n_samples as f64;
        let pt = lerp2(p1, p2, t);
        let d = point_to_ring_distance(pt, boundary);
        distances.push(d);
    }

    let max_dist = distances.iter().cloned().fold(0.0_f64, f64::max);

    // No meaningful variation -> all plateau -> use default distribution
    if max_dist < 0.01 {
        return Vec::new();
    }

    // Adaptive threshold: 30% of max observed distance
    let threshold = max_dist * 0.3;
    let is_plateau: Vec<bool> = distances.iter().map(|&d| d < threshold).collect();

    // If all samples are plateau, return empty
    if is_plateau.iter().all(|&v| v) {
        return Vec::new();
    }

    // Group contiguous plateau regions
    let mut segments = Vec::new();
    let mut start_t: Option<f64> = None;
    for (i, &val) in is_plateau.iter().enumerate() {
        let t = i as f64 / n_samples as f64;
        if val && start_t.is_none() {
            start_t = Some(t);
        } else if !val && start_t.is_some() {
            let end_t = (i - 1) as f64 / n_samples as f64;
            segments.push((start_t.unwrap(), end_t));
            start_t = None;
        }
    }
    if let Some(st) = start_t {
        segments.push((st, 1.0));
    }

    // Apply inset from plateau boundaries
    let inset_t = plateau_inset / edge_len;
    segments
        .iter()
        .filter_map(|&(s, e)| {
            let s2 = s + inset_t;
            let e2 = e - inset_t;
            if e2 > s2 + 1e-6 {
                Some((s2, e2))
            } else {
                None
            }
        })
        .collect()
}

fn point_to_ring_distance(pt: Vec2, ring: &LineString<f64>) -> f64 {
    let mut min_dist = f64::INFINITY;
    for line in ring.lines() {
        let d = point_to_segment_distance(
            pt,
            [line.start.x, line.start.y],
            [line.end.x, line.end.y],
        );
        if d < min_dist {
            min_dist = d;
        }
    }
    min_dist
}

fn point_to_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let vx = b[0] - a[0];
    let vy = b[1] - a[1];
    let l2 = vx * vx + vy * vy;
    if l2 < 1e-12 {
        return dist2(p, a);
    }
    let t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / l2;
    let t = t.clamp(0.0, 1.0);
    let proj = [a[0] + t * vx, a[1] + t * vy];
    dist2(p, proj)
}

// ── Tooth interval computation ──────────────────────────────────────────────

fn intersect_segment_lists(
    segs_a: &[(f64, f64)],
    segs_b: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < segs_a.len() && j < segs_b.len() {
        let (a_s, a_e) = segs_a[i];
        let (b_s, b_e) = segs_b[j];
        let start = a_s.max(b_s);
        let end = a_e.min(b_e);
        if end > start + 1e-6 {
            result.push((start, end));
        }
        if a_e < b_e {
            i += 1;
        } else {
            j += 1;
        }
    }
    result
}

fn reverse_segments(segments: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut result: Vec<(f64, f64)> = segments.iter().map(|&(s, e)| (1.0 - e, 1.0 - s)).collect();
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    result
}

fn compute_n_fingers(edge_len: f64, finger_width: f64, margin: f64) -> usize {
    let usable = edge_len - 2.0 * margin;
    if usable <= 0.0 {
        return 2;
    }
    let n = (usable / finger_width).round().max(1.0) as usize;
    n.max(2)
}

/// Compute the complement of plateau segments: return the gaps BETWEEN the plateaus.
/// Used for through-slots where the plateau detection finds notch valleys but we need
/// the peaks (where material exists) for tooth placement.
fn invert_plateau_segments(plateaus: &[(f64, f64)], inset: f64, edge_len: f64) -> Vec<(f64, f64)> {
    if plateaus.is_empty() {
        return Vec::new();
    }
    let inset_t = if edge_len > 1e-6 { inset / edge_len } else { 0.0 };
    let mut peaks = Vec::new();
    let mut prev_end = 0.0;
    for &(s, e) in plateaus {
        // The plateaus already have inset applied, so their boundaries
        // are slightly inward from the actual valley edges.
        // For the complement (peaks), we use the plateau boundaries directly
        // and apply a small inset to avoid cutting right at the transition.
        if s > prev_end + 1e-6 {
            let peak_s = prev_end + inset_t;
            let peak_e = s - inset_t;
            if peak_e > peak_s + 1e-6 {
                peaks.push((peak_s, peak_e));
            }
        }
        prev_end = e;
    }
    if 1.0 > prev_end + 1e-6 {
        let peak_s = prev_end + inset_t;
        let peak_e = 1.0 - inset_t;
        if peak_e > peak_s + 1e-6 {
            peaks.push((peak_s, peak_e));
        }
    }
    peaks
}

/// Compute parametric exclusion intervals on a line segment (slot_start→slot_end)
/// based on proximity to a set of protected edge segments.  Returns sorted
/// intervals [0..1] that should NOT be cut.
fn compute_protection_intervals(
    slot_start: [f64; 2],
    slot_end: [f64; 2],
    protected_edges: &[([f64; 2], [f64; 2])],
    protect_dist: f64,
) -> Vec<(f64, f64)> {
    let slot_len = dist2(slot_start, slot_end);
    if slot_len < 1e-6 {
        return Vec::new();
    }
    let dx = slot_end[0] - slot_start[0];
    let dy = slot_end[1] - slot_start[1];
    // Unit vector along the slot line
    let ux = dx / slot_len;
    let uy = dy / slot_len;

    let mut exclusions: Vec<(f64, f64)> = Vec::new();

    for &(ep1, ep2) in protected_edges {
        // Project both endpoints of the protected edge onto the slot line
        // and check if the edge is close enough to the slot line to matter.
        let mut t_min = f64::MAX;
        let mut t_max = f64::MIN;
        let mut any_close = false;

        // Sample the protected edge at several points
        let n_samples = 10;
        for i in 0..=n_samples {
            let frac = i as f64 / n_samples as f64;
            let px = ep1[0] + frac * (ep2[0] - ep1[0]);
            let py = ep1[1] + frac * (ep2[1] - ep1[1]);

            // Project onto slot line
            let rel_x = px - slot_start[0];
            let rel_y = py - slot_start[1];
            let t_abs = rel_x * ux + rel_y * uy; // distance along slot line
            let perp = (rel_x * (-uy) + rel_y * ux).abs(); // perpendicular distance

            if perp <= protect_dist * 2.0 {
                any_close = true;
                let t = t_abs / slot_len; // parametric [0..1]
                t_min = t_min.min(t);
                t_max = t_max.max(t);
            }
        }

        if any_close && t_max > t_min - 1e-6 {
            // Expand the exclusion by protect_dist along the slot
            let expand_t = protect_dist / slot_len;
            let excl_start = (t_min - expand_t).max(0.0);
            let excl_end = (t_max + expand_t).min(1.0);
            if excl_end > excl_start + 1e-6 {
                exclusions.push((excl_start, excl_end));
            }
        }
    }

    // Merge overlapping exclusions
    exclusions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for ex in exclusions {
        if let Some(last) = merged.last_mut() {
            if ex.0 <= last.1 + 1e-6 {
                last.1 = last.1.max(ex.1);
                continue;
            }
        }
        merged.push(ex);
    }
    merged
}

/// Compute the complement of exclusion intervals: return the SAFE regions.
fn invert_exclusions(exclusions: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut safe = Vec::new();
    let mut prev_end = 0.0;
    for &(s, e) in exclusions {
        if s > prev_end + 1e-6 {
            safe.push((prev_end, s));
        }
        prev_end = e;
    }
    if 1.0 > prev_end + 1e-6 {
        safe.push((prev_end, 1.0));
    }
    safe
}

fn complement_intervals(
    teeth: &[(f64, f64)],
    plateau_segments: &[(f64, f64)],
    margin_t: f64,
) -> Vec<(f64, f64)> {
    let effective_segments: Vec<(f64, f64)> = if !plateau_segments.is_empty() {
        plateau_segments
            .iter()
            .filter_map(|&(s, e)| {
                let eff_s = s.max(margin_t);
                let eff_e = e.min(1.0 - margin_t);
                if eff_e > eff_s + 1e-6 {
                    Some((eff_s, eff_e))
                } else {
                    None
                }
            })
            .collect()
    } else {
        let s = margin_t;
        let e = 1.0 - margin_t;
        if e > s + 1e-6 {
            vec![(s, e)]
        } else {
            vec![]
        }
    };

    let mut slots = Vec::new();
    for &(seg_s, seg_e) in &effective_segments {
        let teeth_in_seg: Vec<(f64, f64)> = teeth
            .iter()
            .filter_map(|&(t0, t1)| {
                let s = t0.max(seg_s);
                let e = t1.min(seg_e);
                if e > s + 1e-9 {
                    Some((s, e))
                } else {
                    None
                }
            })
            .collect();

        let mut cursor = seg_s;
        for &(t0, t1) in &teeth_in_seg {
            if t0 > cursor + 1e-9 {
                slots.push((cursor, t0));
            }
            cursor = t1;
        }
        if seg_e > cursor + 1e-9 {
            slots.push((cursor, seg_e));
        }
    }

    slots
}

fn build_tooth_intervals(
    edge_len: f64,
    finger_width: f64,
    margin: f64,
    start_with_tooth: bool,
    plateau_segments: &[(f64, f64)],
    min_plateau_length: f64,
    force_odd_segments: bool,
) -> Vec<(f64, f64)> {
    if edge_len < 1e-6 {
        return Vec::new();
    }

    let margin_t = if margin > 0.0 {
        margin / edge_len
    } else {
        0.0
    };

    let segments: Vec<(f64, f64)> = if !plateau_segments.is_empty() {
        plateau_segments
            .iter()
            .filter_map(|&(s, e)| {
                let eff_s = s.max(margin_t);
                let eff_e = e.min(1.0 - margin_t);
                if eff_e > eff_s + 1e-6 {
                    Some((eff_s, eff_e))
                } else {
                    None
                }
            })
            .collect()
    } else {
        vec![(margin_t, 1.0 - margin_t)]
    };

    let mut intervals = Vec::new();
    let start_idx = if start_with_tooth { 0 } else { 1 };

    for (seg_s, seg_e) in &segments {
        let seg_len = (seg_e - seg_s) * edge_len;
        // When plateaus are present, accept segments that can fit at least
        // a partial finger.  Without plateaus, require at least half a finger width.
        let min_len = if !plateau_segments.is_empty() {
            finger_width * 0.5_f64
        } else {
            finger_width * 0.5
        };
        if seg_len < min_len {
            continue;
        }

        let mut n = compute_n_fingers(seg_len, finger_width, 0.0);
        if force_odd_segments && n % 2 == 0 {
            n += 1;
        }

        let mut i = start_idx;
        while i < n {
            let t0 = seg_s + (i as f64 / n as f64) * (seg_e - seg_s);
            let t1 = seg_s + ((i + 1) as f64 / n as f64) * (seg_e - seg_s);
            if t1 > t0 + 1e-9 {
                intervals.push((t0, t1));
            }
            i += 2;
        }
    }

    intervals
}

// ── Outward direction ───────────────────────────────────────────────────────

fn outward_direction(p1: Vec2, p2: Vec2, polygon: &GeoPolygon<f64>) -> Vec2 {
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

    let test1 = geo::Coord {
        x: mid[0] + n1[0] * probe,
        y: mid[1] + n1[1] * probe,
    };
    let test2 = geo::Coord {
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

    // Fallback: centroid heuristic
    let centroid = polygon.exterior().coords().fold([0.0, 0.0], |acc, c| {
        [acc[0] + c.x, acc[1] + c.y]
    });
    let n = polygon.exterior().coords().count() as f64;
    let cx = centroid[0] / n;
    let cy = centroid[1] / n;

    let t1x = mid[0] + n1[0];
    let t1y = mid[1] + n1[1];
    let t2x = mid[0] + n2[0];
    let t2y = mid[1] + n2[1];

    let d1 = (t1x - cx).powi(2) + (t1y - cy).powi(2);
    let d2 = (t2x - cx).powi(2) + (t2y - cy).powi(2);

    if d1 > d2 {
        n1
    } else {
        n2
    }
}

// ── Comb / tooth geometry ───────────────────────────────────────────────────

fn make_tooth_rect(
    p1: Vec2,
    p2: Vec2,
    t0: f64,
    t1: f64,
    depth: f64,
    outward: Vec2,
    overlap: f64,
) -> GeoPolygon<f64> {
    let s0 = lerp2(p1, p2, t0);
    let s1 = lerp2(p1, p2, t1);
    let nx = outward[0];
    let ny = outward[1];

    let coords = vec![
        Coord {
            x: s0[0] - nx * overlap,
            y: s0[1] - ny * overlap,
        },
        Coord {
            x: s1[0] - nx * overlap,
            y: s1[1] - ny * overlap,
        },
        Coord {
            x: s1[0] + nx * depth,
            y: s1[1] + ny * depth,
        },
        Coord {
            x: s0[0] + nx * depth,
            y: s0[1] + ny * depth,
        },
        Coord {
            x: s0[0] - nx * overlap,
            y: s0[1] - ny * overlap,
        },
    ];

    GeoPolygon::new(LineString::new(coords), vec![])
}

fn make_comb(
    p1: Vec2,
    p2: Vec2,
    depth: f64,
    outward: Vec2,
    tooth_intervals: &[(f64, f64)],
    overlap: f64,
) -> Vec<GeoPolygon<f64>> {
    tooth_intervals
        .iter()
        .filter_map(|&(t0, t1)| {
            let rect = make_tooth_rect(p1, p2, t0, t1, depth, outward, overlap);
            if rect.unsigned_area() > 0.0 {
                Some(rect)
            } else {
                None
            }
        })
        .collect()
}

fn make_strip(
    p1: Vec2,
    p2: Vec2,
    depth: f64,
    outward: Vec2,
    margin: f64,
    edge_len: f64,
    plateau_segments: &[(f64, f64)],
    min_plateau_length: f64,
    overlap: f64,
) -> Vec<GeoPolygon<f64>> {
    if edge_len < 1e-6 {
        return Vec::new();
    }

    let margin_t = if margin > 0.0 {
        margin / edge_len
    } else {
        0.0
    };

    let segments: Vec<(f64, f64)> = if !plateau_segments.is_empty() {
        plateau_segments
            .iter()
            .filter_map(|&(s, e)| {
                let eff_s = s.max(margin_t);
                let eff_e = e.min(1.0 - margin_t);
                if eff_e > eff_s + 1e-6 {
                    Some((eff_s, eff_e))
                } else {
                    None
                }
            })
            .collect()
    } else {
        vec![(margin_t, 1.0 - margin_t)]
    };

    let nx = outward[0];
    let ny = outward[1];

    segments
        .iter()
        .filter_map(|&(seg_s, seg_e)| {
            let seg_len = (seg_e - seg_s) * edge_len;
            if seg_len < min_plateau_length.max(1e-3) {
                return None;
            }

            let s0 = lerp2(p1, p2, seg_s);
            let s1 = lerp2(p1, p2, seg_e);
            let coords = vec![
                Coord {
                    x: s0[0] - nx * overlap,
                    y: s0[1] - ny * overlap,
                },
                Coord {
                    x: s1[0] - nx * overlap,
                    y: s1[1] - ny * overlap,
                },
                Coord {
                    x: s1[0] + nx * depth,
                    y: s1[1] + ny * depth,
                },
                Coord {
                    x: s0[0] + nx * depth,
                    y: s0[1] + ny * depth,
                },
                Coord {
                    x: s0[0] - nx * overlap,
                    y: s0[1] - ny * overlap,
                },
            ];
            let rect = GeoPolygon::new(LineString::new(coords), vec![]);
            if rect.unsigned_area() > 0.0 {
                Some(rect)
            } else {
                None
            }
        })
        .collect()
}

// ── Edge matching helpers ───────────────────────────────────────────────────

pub fn find_matching_edge_index(proj: &Projection2D, shared_edge: &SharedEdge) -> Option<usize> {
    let tol = 0.5;
    for (idx, edge_3d) in proj.edge_map_3d.iter().enumerate() {
        for se in [&shared_edge.edge_a, &shared_edge.edge_b] {
            let mid_match = points_close3(edge_3d.midpoint, se.midpoint, tol);
            if mid_match {
                return Some(idx);
            }

            let fwd = points_close3(edge_3d.start, se.start, tol)
                && points_close3(edge_3d.end, se.end, tol);
            let rev = points_close3(edge_3d.start, se.end, tol)
                && points_close3(edge_3d.end, se.start, tol);
            if fwd || rev {
                return Some(idx);
            }
        }
    }
    None
}

fn edges_reversed(
    proj_a: &Projection2D,
    edge_idx_a: usize,
    proj_b: &Projection2D,
    edge_idx_b: usize,
    tol: f64,
) -> bool {
    let ea = &proj_a.edge_map_3d[edge_idx_a];
    let eb = &proj_b.edge_map_3d[edge_idx_b];

    let fwd = points_close3(ea.start, eb.start, tol) && points_close3(ea.end, eb.end, tol);
    let rev = points_close3(ea.start, eb.end, tol) && points_close3(ea.end, eb.start, tol);

    rev && !fwd
}

pub fn find_bottom_edge_endpoints(
    wall: &PlanarFace,
    bottom: &PlanarFace,
    tol: f64,
) -> Option<(Vec3, Vec3)> {
    let bn = normalize3(bottom.normal);

    let plane_dist = |pt: Vec3| -> f64 { dot3(bn, sub3(pt, bottom.center)) };

    let edge_dists: Vec<(f64, &EdgeData)> = wall
        .outer_wire_edges
        .iter()
        .map(|e| (plane_dist(e.midpoint).abs(), e))
        .collect();

    let min_dist = edge_dists
        .iter()
        .map(|(d, _)| *d)
        .fold(f64::INFINITY, f64::min);
    let bottom_edges: Vec<&EdgeData> = edge_dists
        .iter()
        .filter(|(d, _)| *d < min_dist + tol)
        .map(|(_, e)| *e)
        .collect();

    if bottom_edges.is_empty() {
        return None;
    }

    let all_pts: Vec<Vec3> = bottom_edges
        .iter()
        .flat_map(|e| vec![e.start, e.end])
        .collect();

    let mut max_d = 0.0;
    let mut best_pair = None;
    for i in 0..all_pts.len() {
        for j in (i + 1)..all_pts.len() {
            let d = dist3(all_pts[i], all_pts[j]);
            if d > max_d {
                max_d = d;
                best_pair = Some((all_pts[i], all_pts[j]));
            }
        }
    }

    best_pair
}

// ── Main finger joint application ───────────────────────────────────────────

pub struct FingerJointParams {
    pub finger_width: f64,
    pub edge_margin: f64,
    pub notch_buffer: f64,
    pub plateau_inset: f64,
    pub min_plateau_length: f64,
    pub kerf: f64,
    pub preserve_outer_dims: bool,
}

impl Default for FingerJointParams {
    fn default() -> Self {
        Self {
            finger_width: DEFAULT_FINGER_WIDTH,
            edge_margin: DEFAULT_EDGE_MARGIN,
            notch_buffer: DEFAULT_NOTCH_BUFFER,
            plateau_inset: DEFAULT_PLATEAU_INSET,
            min_plateau_length: DEFAULT_MIN_PLATEAU_LENGTH,
            kerf: 0.0,
            preserve_outer_dims: true,
        }
    }
}

pub fn apply_finger_joints(
    projections: &HashMap<usize, Projection2D>,
    shared_edges: &[SharedEdge],
    bottom_id: usize,
    thickness: f64,
    params: &FingerJointParams,
    faces: Option<&[PlanarFace]>,
) -> (HashMap<usize, Vec<Vec2>>, HashMap<usize, Vec<Vec<Vec2>>>) {
    let kerf_half = params.kerf / 2.0;

    // Convert projections to geo polygons
    let mut shapes: HashMap<usize, GeoPolygon<f64>> = HashMap::new();
    let mut raw_shapes: HashMap<usize, GeoPolygon<f64>> = HashMap::new();
    for (&fid, proj) in projections {
        let shape = polygon_to_geo(&proj.outer_polygon, &proj.inner_polygons);
        raw_shapes.insert(fid, shape.clone());
        shapes.insert(fid, shape);
    }

    // Find bottom-adjacent faces
    let mut bottom_adjacent: HashSet<usize> = HashSet::new();
    for se in shared_edges {
        if se.face_a_id == bottom_id {
            bottom_adjacent.insert(se.face_b_id);
        } else if se.face_b_id == bottom_id {
            bottom_adjacent.insert(se.face_a_id);
        }
    }

    // Build a map of bottom-plate edge endpoints for corner detection.
    // For each bottom-wall shared edge, store the 2D endpoints on the
    // bottom plate.  An endpoint that appears in multiple edges is a
    // "corner" where two walls meet the bottom plate.
    let bottom_proj_ref = projections.get(&bottom_id);
    let mut bottom_edge_endpoints: Vec<([f64; 2], [f64; 2], usize)> = Vec::new(); // (p1,p2,wall_face_id)
    for se in shared_edges {
        let wall_id = if se.face_a_id == bottom_id {
            se.face_b_id
        } else if se.face_b_id == bottom_id {
            se.face_a_id
        } else {
            continue;
        };
        if !bottom_adjacent.contains(&wall_id) {
            continue;
        }
        if let Some(bp) = bottom_proj_ref {
            if let Some(edge_idx) = find_matching_edge_index(bp, se) {
                let (ep1, ep2) = bp.outer_edges_2d[edge_idx];
                bottom_edge_endpoints.push((ep1, ep2, wall_id));
            }
        }
    }

    // For each bottom-plate edge, determine if p1 or p2 is a "corner"
    // (shared with another bottom-plate edge endpoint within tolerance).
    let corner_tol = 1.0; // mm
    let is_corner_point = |pt: [f64; 2], own_wall_id: usize| -> bool {
        for &(ep1, ep2, wid) in &bottom_edge_endpoints {
            if wid == own_wall_id {
                continue;
            }
            if dist2(pt, ep1) < corner_tol || dist2(pt, ep2) < corner_tol {
                return true;
            }
        }
        false
    };

    // Apply finger joints along shared edges
    for se in shared_edges {
        let fid_a = se.face_a_id;
        let fid_b = se.face_b_id;

        if !projections.contains_key(&fid_a) || !projections.contains_key(&fid_b) {
            continue;
        }

        // Determine positive/negative
        let (pos_id, neg_id) = if fid_a == bottom_id {
            (fid_a, fid_b)
        } else if fid_b == bottom_id {
            (fid_b, fid_a)
        } else {
            let a_adj = bottom_adjacent.contains(&fid_a);
            let b_adj = bottom_adjacent.contains(&fid_b);
            if a_adj != b_adj {
                if a_adj {
                    (fid_b, fid_a)
                } else {
                    (fid_a, fid_b)
                }
            } else {
                (fid_a.min(fid_b), fid_a.max(fid_b))
            }
        };

        let is_bottom_pair = fid_a == bottom_id || fid_b == bottom_id;
        let preserve_on_this_pair = params.preserve_outer_dims && is_bottom_pair;

        let pos_proj = &projections[&pos_id];
        let neg_proj = &projections[&neg_id];
        let pos_edge_idx = match find_matching_edge_index(pos_proj, se) {
            Some(idx) => idx,
            None => continue,
        };
        let neg_edge_idx = match find_matching_edge_index(neg_proj, se) {
            Some(idx) => idx,
            None => continue,
        };

        let (pos_p1, pos_p2) = pos_proj.outer_edges_2d[pos_edge_idx];
        let (neg_p1, neg_p2) = neg_proj.outer_edges_2d[neg_edge_idx];

        let pos_edge_len = dist2(pos_p1, pos_p2);
        let neg_edge_len = dist2(neg_p1, neg_p2);
        if pos_edge_len < 1e-6 || neg_edge_len < 1e-6 {
            continue;
        }

        // Detect plateau segments
        let pos_plateaus = find_plateau_segments(
            pos_p1,
            pos_p2,
            &raw_shapes[&pos_id],
            0.5,
            params.plateau_inset,
        );
        let neg_plateaus = find_plateau_segments(
            neg_p1,
            neg_p2,
            &raw_shapes[&neg_id],
            0.5,
            params.plateau_inset,
        );

        // Determine shared plateau segments
        let reversed_edge =
            edges_reversed(pos_proj, pos_edge_idx, neg_proj, neg_edge_idx, 0.5);

        let (shared_pos, shared_neg) = if !pos_plateaus.is_empty() && !neg_plateaus.is_empty() {
            let neg_in_pos_space = if reversed_edge {
                reverse_segments(&neg_plateaus)
            } else {
                neg_plateaus.clone()
            };
            let sp = intersect_segment_lists(&pos_plateaus, &neg_in_pos_space);
            let sn = if reversed_edge {
                reverse_segments(&sp)
            } else {
                sp.clone()
            };
            (sp, sn)
        } else if !pos_plateaus.is_empty() {
            let sn = if reversed_edge {
                reverse_segments(&pos_plateaus)
            } else {
                pos_plateaus.clone()
            };
            (pos_plateaus, sn)
        } else if !neg_plateaus.is_empty() {
            let sp = if reversed_edge {
                reverse_segments(&neg_plateaus)
            } else {
                neg_plateaus.clone()
            };
            (sp, neg_plateaus)
        } else {
            (Vec::new(), Vec::new())
        };

        // Build tooth intervals.
        //
        // For bottom-plate edges (preserve_outer_dims), reduce the margin
        // at endpoints that are "corners" (shared with another bottom-wall edge).
        // This ensures finger joints extend to the corner point instead of
        // leaving a gap.
        let effective_margin = if is_bottom_pair {
            let p1_corner = is_corner_point(pos_p1, neg_id);
            let p2_corner = is_corner_point(pos_p2, neg_id);
            if p1_corner && p2_corner {
                0.0 // Both ends are corners - use no margin
            } else if p1_corner || p2_corner {
                // One end is a corner: use half margin to shift fingers toward corner.
                // The actual zero-margin extension happens below via interval extension.
                params.edge_margin * 0.5
            } else {
                params.edge_margin
            }
        } else {
            params.edge_margin
        };

        let mut tooth_pos = build_tooth_intervals(
            pos_edge_len,
            params.finger_width,
            effective_margin,
            true,
            &shared_pos,
            params.min_plateau_length,
            true,
        );

        // Extend first/last tooth intervals to cover corner gaps.
        // At corners, the finger joints should extend to the edge endpoint.
        if is_bottom_pair && !tooth_pos.is_empty() {
            let p1_corner = is_corner_point(pos_p1, neg_id);
            let p2_corner = is_corner_point(pos_p2, neg_id);
            let min_margin_t = 1.0 / pos_edge_len.max(1.0); // ~1mm minimum

            if p1_corner {
                // Extend first tooth toward t=0 (start of edge)
                tooth_pos[0].0 = min_margin_t.min(tooth_pos[0].0);
            }
            if p2_corner {
                // Extend last tooth toward t=1 (end of edge)
                let last = tooth_pos.len() - 1;
                tooth_pos[last].1 = (1.0 - min_margin_t).max(tooth_pos[last].1);
            }
        }

        let tooth_neg = if reversed_edge {
            reverse_segments(&tooth_pos)
        } else {
            tooth_pos.clone()
        };

        // Process both faces
        for &(face_id, is_positive, p1, p2, ref plateaus, ref teeth) in &[
            (pos_id, true, pos_p1, pos_p2, &shared_pos, &tooth_pos),
            (neg_id, false, neg_p1, neg_p2, &shared_neg, &tooth_neg),
        ] {
            let shape = shapes.get(&face_id).unwrap().clone();
            let outward_dir = outward_direction(p1, p2, &shape);
            let depth = thickness;
            if depth <= 0.0 {
                continue;
            }

            if is_positive {
                if preserve_on_this_pair {
                    // Preserve outer dimensions: subtract only the slot regions
                    // (complement of tooth intervals) inward from the edge.
                    // This avoids a fragile union step.
                    let inward = [-outward_dir[0], -outward_dir[1]];
                    let edge_len = dist2(p1, p2);
                    let eff_margin_t = if effective_margin > 0.0 && edge_len > 1e-6 {
                        effective_margin / edge_len
                    } else {
                        0.0
                    };

                    let slot_intervals =
                        complement_intervals(teeth, plateaus, eff_margin_t);
                    let slot_rects =
                        make_comb(p1, p2, depth, inward, &slot_intervals, 0.5);
                    if !slot_rects.is_empty() {
                        let new_shape = boolean_difference_multi(&shape, &slot_rects);
                        shapes.insert(face_id, new_shape);
                    }
                } else {
                    let comb_teeth = make_comb(p1, p2, depth, outward_dir, teeth, 0.5);
                    if !comb_teeth.is_empty() {
                        let new_shape = boolean_union_multi(&shape, &comb_teeth);
                        shapes.insert(face_id, new_shape);
                    }
                }
            } else {
                let inward = [-outward_dir[0], -outward_dir[1]];
                let slot_teeth = make_comb(p1, p2, depth, inward, teeth, 0.5);
                if !slot_teeth.is_empty() {
                    let new_shape = boolean_difference_multi(&shape, &slot_teeth);
                    shapes.insert(face_id, new_shape);
                }
            }
        }
    }

    // Apply through-slot joints
    if let Some(all_faces) = faces {
        let face_map: HashMap<usize, &PlanarFace> =
            all_faces.iter().map(|f| (f.face_id, f)).collect();

        if let Some(bottom_face) = face_map.get(&bottom_id) {
            let bottom_proj = &projections[&bottom_id];

            // Build protected bottom joint zones: 2D edge segments on the
            // bottom plate where direct wall-bottom finger joints exist.
            // Through-slot cuts must stay away from these areas to avoid
            // interfering with corner finger joints.
            let protect_dist = params.notch_buffer.max(0.5) + thickness.max(0.0);
            let mut protected_bottom_edges: Vec<([f64; 2], [f64; 2])> = Vec::new();
            for se in shared_edges {
                let other = if se.face_a_id == bottom_id {
                    se.face_b_id
                } else if se.face_b_id == bottom_id {
                    se.face_a_id
                } else {
                    continue;
                };
                if !bottom_adjacent.contains(&other) {
                    continue;
                }
                if let Some(edge_idx) = find_matching_edge_index(bottom_proj, se) {
                    let (ep1, ep2) = bottom_proj.outer_edges_2d[edge_idx];
                    protected_bottom_edges.push((ep1, ep2));
                }
            }

            for (&fid, proj) in projections {
                if fid == bottom_id || bottom_adjacent.contains(&fid) {
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

                // Through-slots on the bottom plate
                let slot_start = project_point_3d_to_2d(
                    p_start_3d,
                    bottom_proj.origin_3d,
                    bottom_proj.u_axis,
                    bottom_proj.v_axis,
                );
                let slot_end = project_point_3d_to_2d(
                    p_end_3d,
                    bottom_proj.origin_3d,
                    bottom_proj.u_axis,
                    bottom_proj.v_axis,
                );
                let slot_len = dist2(slot_start, slot_end);
                if slot_len < 1e-6 {
                    continue;
                }

                let bottom_shape = shapes.get(&bottom_id).unwrap().clone();
                let inward = {
                    let out = outward_direction(slot_start, slot_end, &bottom_shape);
                    [-out[0], -out[1]]
                };

                // Wall projection
                let wall_start = project_point_3d_to_2d(
                    p_start_3d,
                    proj.origin_3d,
                    proj.u_axis,
                    proj.v_axis,
                );
                let wall_end = project_point_3d_to_2d(
                    p_end_3d,
                    proj.origin_3d,
                    proj.u_axis,
                    proj.v_axis,
                );
                let wall_edge_len = dist2(wall_start, wall_end);
                if wall_edge_len < 1e-6 {
                    continue;
                }

                // Plateau detection for through-slots
                //
                // The bottom plate's raw shape may have notch markers from the 3D
                // model where the back wall connects.  find_plateau_segments treats
                // points close to the polygon boundary as "plateaus", so the notch
                // VALLEYS (where the boundary dips toward the through-slot line) are
                // returned as plateaus.  We actually need the PEAKS (where material
                // exists between the notch valleys) for tooth placement.
                //
                // Strategy: detect the valleys on the raw bottom shape, then INVERT
                // them to get peak segments where through-slot teeth should go.
                let wall_plateaus = find_plateau_segments(
                    wall_start,
                    wall_end,
                    &raw_shapes[&fid],
                    0.5,
                    params.plateau_inset,
                );
                let bottom_valley_plateaus = find_plateau_segments(
                    slot_start,
                    slot_end,
                    &raw_shapes[&bottom_id],
                    0.5,
                    params.plateau_inset,
                );

                // Invert bottom plateaus: valleys → peaks (where material exists)
                let bottom_peaks = invert_plateau_segments(
                    &bottom_valley_plateaus,
                    params.plateau_inset,
                    slot_len,
                );

                let shared_plateaus = if !wall_plateaus.is_empty() && !bottom_peaks.is_empty() {
                    let intersection =
                        intersect_segment_lists(&wall_plateaus, &bottom_peaks);
                    let min_useful = (params.finger_width * 0.5).max(params.min_plateau_length);
                    let filtered: Vec<(f64, f64)> = intersection
                        .into_iter()
                        .filter(|(s, e)| (e - s) * wall_edge_len >= min_useful - 1e-6)
                        .collect();
                    if filtered.is_empty() {
                        bottom_peaks
                    } else {
                        filtered
                    }
                } else if !bottom_peaks.is_empty() {
                    bottom_peaks
                } else if !wall_plateaus.is_empty() {
                    wall_plateaus
                } else {
                    Vec::new()
                };

                let mut shared_tooth_intervals = build_tooth_intervals(
                    slot_len,
                    params.finger_width,
                    params.edge_margin,
                    false,
                    &shared_plateaus,
                    params.min_plateau_length,
                    true,
                );

                // Fallback: if plateau segments are too narrow for teeth,
                // retry with no plateaus (full edge with margin).
                if shared_tooth_intervals.is_empty() && !shared_plateaus.is_empty() {
                    shared_tooth_intervals = build_tooth_intervals(
                        slot_len,
                        params.finger_width,
                        params.edge_margin,
                        false,
                        &[],  // no plateau restriction
                        params.min_plateau_length,
                        true,
                    );
                }

                // Compute protection zones: parametric intervals on the
                // through-slot line that overlap with direct bottom-wall
                // seam edges.  Through-slot cuts must avoid these regions
                // to preserve the corner finger joints.
                let exclusions = compute_protection_intervals(
                    slot_start, slot_end, &protected_bottom_edges, protect_dist,
                );
                let safe_intervals = invert_exclusions(&exclusions);

                // Filter tooth intervals to stay within safe zones
                let safe_tooth_intervals: Vec<(f64, f64)> = if !exclusions.is_empty() {
                    intersect_segment_lists(&shared_tooth_intervals, &safe_intervals)
                } else {
                    shared_tooth_intervals.clone()
                };

                // Cut slots in bottom (only in safe zones)
                let slot_teeth =
                    make_comb(slot_start, slot_end, thickness, inward, &safe_tooth_intervals, 0.01);
                if !slot_teeth.is_empty() {
                    let bottom_shape = shapes.get(&bottom_id).unwrap().clone();
                    let new_shape = boolean_difference_multi(&bottom_shape, &slot_teeth);
                    shapes.insert(bottom_id, new_shape);
                }

                // Cut outward strip along the through-slot edge, but only in
                // safe zones (not overlapping direct bottom-wall seam edges).
                // This cleans up 3D model notch markers while preserving
                // corner finger joints.
                let outward_bottom = [-inward[0], -inward[1]];
                let edge_notch_strip = make_comb(
                    slot_start,
                    slot_end,
                    thickness * 3.0,
                    outward_bottom,
                    &safe_intervals,
                    0.01,
                );
                if !edge_notch_strip.is_empty() {
                    let bottom_shape = shapes.get(&bottom_id).unwrap().clone();
                    let new_shape = boolean_difference_multi(&bottom_shape, &edge_notch_strip);
                    shapes.insert(bottom_id, new_shape);
                }

                // Add tabs on wall (matching safe tooth intervals from bottom)
                let wall_shape = shapes.get(&fid).unwrap().clone();
                let outward_wall = outward_direction(wall_start, wall_end, &wall_shape);
                let wall_teeth = make_comb(
                    wall_start,
                    wall_end,
                    thickness,
                    outward_wall,
                    &safe_tooth_intervals,
                    0.5,
                );
                if !wall_teeth.is_empty() {
                    let new_shape = boolean_union_multi(&wall_shape, &wall_teeth);
                    shapes.insert(fid, new_shape);
                }
            }
        }
    }

    // Apply kerf compensation
    if kerf_half.abs() > 1e-9 {
        for (_fid, shape) in shapes.iter_mut() {
            *shape = buffer_polygon(shape, kerf_half);
        }
    }

    // Convert back to vertex lists
    let mut modified = HashMap::new();
    let mut slot_cutouts = HashMap::new();

    for (&fid, shape) in &shapes {
        let (outer, inners) = geo_to_vertices(shape);
        modified.insert(fid, outer);
        slot_cutouts.insert(fid, inners);
    }

    (modified, slot_cutouts)
}
