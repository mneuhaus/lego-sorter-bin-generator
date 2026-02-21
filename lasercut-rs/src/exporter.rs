use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::layout::{compute_arrangement, prepare_parts, ArrangedPart, PreparedPart};
use crate::types::*;

pub const DEFAULT_SHEET_WIDTH_MM: f64 = 400.0;
pub const DEFAULT_SHEET_HEIGHT_MM: f64 = 800.0;

fn format_thickness_mm(thickness: f64) -> String {
    format!("{:.3}", thickness)
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn rotate_geoms_90_cw(geoms: &[ArrangedPart], width: f64) -> Vec<ArrangedPart> {
    geoms
        .iter()
        .map(|g| {
            let poly = g.polygon.iter().map(|p| [p[1], width - p[0]]).collect();
            let holes = g
                .holes
                .iter()
                .map(|h| h.iter().map(|p| [p[1], width - p[0]]).collect())
                .collect();
            ArrangedPart {
                fid: g.fid,
                label: g.label.clone(),
                polygon: poly,
                holes,
            }
        })
        .collect()
}

// ── SVG export ──────────────────────────────────────────────────────────────

pub fn export_svg(
    projections: &HashMap<usize, Projection2D>,
    modified_polygons: &HashMap<usize, Vec<Vec2>>,
    output_path: &str,
    slot_cutouts: &HashMap<usize, Vec<Vec<Vec2>>>,
    layout: &str,
    wall_offset: f64,
    shared_edges: &[SharedEdge],
    bottom_id: usize,
    faces: Option<&[PlanarFace]>,
    thickness: f64,
    sheet_width: f64,
    sheet_height: f64,
    padding: f64,
) -> anyhow::Result<String> {
    let output = Path::new(output_path);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let layout_mode = if layout == "folded" { "unfolded" } else { layout };
    let parts = prepare_parts(projections, modified_polygons, slot_cutouts);

    let (mut geoms, mut actual_w, mut actual_h) = compute_arrangement(
        &parts,
        layout_mode,
        projections,
        shared_edges,
        bottom_id,
        faces,
        padding,
        wall_offset,
        Some(sheet_width),
    );

    // Rotate if needed to fit
    if actual_w > sheet_width || actual_h > sheet_height {
        if actual_h <= sheet_width && actual_w <= sheet_height {
            geoms = rotate_geoms_90_cw(&geoms, actual_w);
            std::mem::swap(&mut actual_w, &mut actual_h);
        }
    }

    if actual_w > sheet_width || actual_h > sheet_height {
        anyhow::bail!(
            "Layout does not fit target sheet: needed {:.1}x{:.1} mm, sheet is {:.1}x{:.1} mm",
            actual_w,
            actual_h,
            sheet_width,
            sheet_height,
        );
    }

    // Write SVG
    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{:.1}mm" height="{:.1}mm" viewBox="0 0 {:.1} {:.1}">
<desc>Lasercut Layout - Thickness: {} mm</desc>
"#,
        sheet_width,
        sheet_height,
        sheet_width,
        sheet_height,
        format_thickness_mm(thickness),
    ));

    let stroke_width = 0.5;
    for g in &geoms {
        svg.push_str(&format!(r#"<g id="{}">"#, g.label));
        svg.push('\n');

        if g.polygon.len() >= 2 {
            svg.push_str(r#"<polygon points=""#);
            for (i, p) in g.polygon.iter().enumerate() {
                if i > 0 {
                    svg.push(' ');
                }
                svg.push_str(&format!("{:.4},{:.4}", p[0], p[1]));
            }
            svg.push_str(&format!(
                r#"" fill="none" stroke="black" stroke-width="{}" stroke-linejoin="round" stroke-linecap="butt"/>"#,
                stroke_width
            ));
            svg.push('\n');
        }

        for hole in &g.holes {
            if hole.len() >= 2 {
                svg.push_str(r#"<polygon points=""#);
                for (i, p) in hole.iter().enumerate() {
                    if i > 0 {
                        svg.push(' ');
                    }
                    svg.push_str(&format!("{:.4},{:.4}", p[0], p[1]));
                }
                svg.push_str(&format!(
                    r#"" fill="none" stroke="black" stroke-width="{}" stroke-linejoin="round" stroke-linecap="butt"/>"#,
                    stroke_width
                ));
                svg.push('\n');
            }
        }

        svg.push_str("</g>\n");
    }

    svg.push_str("</svg>\n");

    std::fs::write(output_path, &svg)?;
    Ok(output_path.to_string())
}

// ── DXF export ──────────────────────────────────────────────────────────────

pub fn export_dxf(
    projections: &HashMap<usize, Projection2D>,
    modified_polygons: &HashMap<usize, Vec<Vec2>>,
    output_path: &str,
    per_face: bool,
    slot_cutouts: &HashMap<usize, Vec<Vec<Vec2>>>,
    layout: &str,
    wall_offset: f64,
    shared_edges: &[SharedEdge],
    bottom_id: usize,
    faces: Option<&[PlanarFace]>,
    thickness: f64,
    padding: f64,
) -> anyhow::Result<Vec<String>> {
    let output = Path::new(output_path);
    let mut written = Vec::new();

    if per_face {
        std::fs::create_dir_all(output_path)?;
        for (&fid, polygon) in modified_polygons {
            let proj = &projections[&fid];
            let label = if proj.label.is_empty() {
                format!("face_{}", fid)
            } else {
                proj.label.clone()
            };
            let filepath = format!("{}/{}.dxf", output_path, label);
            let cutouts = slot_cutouts.get(&fid).cloned().unwrap_or_default();
            write_dxf_file(&filepath, &[(polygon.clone(), cutouts, label.clone())], thickness)?;
            written.push(filepath);
        }
    } else {
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let layout_mode = if layout == "folded" { "unfolded" } else { layout };
        let parts = prepare_parts(projections, modified_polygons, slot_cutouts);
        let (geoms, _, _) = compute_arrangement(
            &parts,
            layout_mode,
            projections,
            shared_edges,
            bottom_id,
            faces,
            padding,
            wall_offset,
            None,
        );

        let entities: Vec<(Vec<Vec2>, Vec<Vec<Vec2>>, String)> = geoms
            .iter()
            .map(|g| (g.polygon.clone(), g.holes.clone(), g.label.clone()))
            .collect();

        write_dxf_file(output_path, &entities, thickness)?;
        written.push(output_path.to_string());
    }

    Ok(written)
}

fn write_dxf_file(
    filepath: &str,
    entities: &[(Vec<Vec2>, Vec<Vec<Vec2>>, String)],
    _thickness: f64,
) -> anyhow::Result<()> {
    let mut drawing = dxf::Drawing::new();
    drawing.header.version = dxf::enums::AcadVersion::R2000;

    for (polygon, holes, layer_name) in entities {
        if polygon.len() >= 2 {
            let mut lwp = dxf::entities::LwPolyline::default();
            // flags bit 0 = closed
            lwp.flags = 1;
            for p in polygon {
                lwp.vertices.push(dxf::LwPolylineVertex {
                    x: p[0],
                    y: p[1],
                    ..Default::default()
                });
            }
            let mut entity =
                dxf::entities::Entity::new(dxf::entities::EntityType::LwPolyline(lwp));
            entity.common.layer = layer_name.clone();
            drawing.add_entity(entity);
        }

        for hole in holes {
            if hole.len() >= 2 {
                let mut lwp = dxf::entities::LwPolyline::default();
                lwp.flags = 1;
                for p in hole {
                    lwp.vertices.push(dxf::LwPolylineVertex {
                        x: p[0],
                        y: p[1],
                        ..Default::default()
                    });
                }
                let mut entity =
                    dxf::entities::Entity::new(dxf::entities::EntityType::LwPolyline(lwp));
                entity.common.layer = layer_name.clone();
                drawing.add_entity(entity);
            }
        }
    }

    drawing.save_file(filepath)?;
    Ok(())
}
