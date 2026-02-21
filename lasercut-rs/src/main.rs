use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use clap::Parser;

mod types;
mod math_utils;
mod step_loader;
mod face_classifier;
mod projector;
mod finger_joints;
mod layout;
mod exporter;
mod verification;

use types::*;
use finger_joints::FingerJointParams;

#[derive(Parser, Debug)]
#[command(name = "lasercut")]
#[command(about = "Generate laser-cut DXF/SVG files from STEP models with finger joints")]
struct Cli {
    /// Path to STEP file
    input: String,

    /// Material thickness in mm
    #[arg(long, default_value_t = 3.0)]
    thickness: f64,

    /// Laser kerf in mm for compensation
    #[arg(long, default_value_t = 0.0)]
    kerf: f64,

    /// Explicit kerf sweep values in mm
    #[arg(long, num_args = 1..)]
    kerfs: Option<Vec<f64>>,

    /// Generate a kerf sweep
    #[arg(long)]
    kerf_range: bool,

    /// Kerf sweep start in mm
    #[arg(long, default_value_t = 0.0)]
    kerf_start: f64,

    /// Kerf sweep end in mm
    #[arg(long, default_value_t = 0.10)]
    kerf_end: f64,

    /// Kerf sweep step in mm
    #[arg(long, default_value_t = 0.02)]
    kerf_step: f64,

    /// Finger width in mm, 0 = auto
    #[arg(long, default_value_t = 0.0)]
    finger_width: f64,

    /// Output formats
    #[arg(long, default_values_t = vec!["dxf".to_string(), "svg".to_string()])]
    format: Vec<String>,

    /// Output directory
    #[arg(long, default_value = "./output")]
    output: String,

    /// Minimum face area in mm²
    #[arg(long, default_value_t = 100.0)]
    min_area: f64,

    /// Write one DXF file per face
    #[arg(long)]
    per_face: bool,

    /// Which body to process (0-based, -1 for all)
    #[arg(long, default_value_t = 0)]
    body: i32,

    /// Safe zone at edge ends in mm
    #[arg(long, default_value_t = -1.0)]
    edge_margin: f64,

    /// Safe zone around notches in mm
    #[arg(long, default_value_t = -1.0)]
    notch_buffer: f64,

    /// Inset from plateau boundaries in mm
    #[arg(long, default_value_t = -1.0)]
    plateau_inset: f64,

    /// Minimum plateau segment length in mm
    #[arg(long, default_value_t = -1.0)]
    min_plateau_length: f64,

    /// Part layout mode
    #[arg(long, default_value = "unfolded")]
    layout: String,

    /// Gap from bottom to walls in unfolded layout
    #[arg(long, default_value_t = 10.0)]
    wall_offset: f64,

    /// SVG sheet width in mm
    #[arg(long, default_value_t = 400.0)]
    sheet_width: f64,

    /// SVG sheet height in mm
    #[arg(long, default_value_t = 800.0)]
    sheet_height: f64,

    /// Run 2D seam complement verification
    #[arg(long)]
    verify: bool,

    /// Run tab-overlap interference proxy checks
    #[arg(long)]
    verify_3d: bool,

    /// Verification sample step in mm
    #[arg(long, default_value_t = 0.25)]
    verify_step: f64,

    /// Allowed seam mismatch ratio
    #[arg(long, default_value_t = 0.02)]
    verify_tolerance: f64,

    /// Allowed interference ratio
    #[arg(long, default_value_t = 0.01)]
    verify_interference_tolerance: f64,

    /// Write per-joint debug SVGs
    #[arg(long)]
    verify_debug: bool,

    /// Exit with non-zero status if verification fails
    #[arg(long)]
    verify_strict: bool,
}

fn format_mm(value: f64) -> String {
    format!("{:.3}", value)
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn cut_settings_suffix(thickness: f64, kerf: f64) -> String {
    format!("_{}mm_kerf_{}mm", format_mm(thickness), format_mm(kerf))
}

fn settings_output_dir(root: &str, thickness: f64, kerf: f64, layout: &str) -> String {
    format!(
        "{}/{}mm/kerf_{}mm/{}",
        root,
        format_mm(thickness),
        format_mm(kerf),
        layout
    )
}

fn kerf_sweep(start: f64, end: f64, step: f64) -> Vec<f64> {
    let mut vals = Vec::new();
    let mut k = start;
    while k <= end + 1e-9 {
        vals.push((k * 1e6).round() / 1e6);
        k += step;
    }
    if let Some(&last) = vals.last() {
        if (last - end).abs() > 1e-6 && last < end {
            vals.push((end * 1e6).round() / 1e6);
        }
    }
    vals
}

fn normalize_layout(layout: &str) -> &str {
    if layout == "folded" {
        "unfolded"
    } else {
        layout
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if !Path::new(&cli.input).exists() {
        eprintln!("Error: Input file not found: {}", cli.input);
        std::process::exit(1);
    }

    let fw = if cli.finger_width > 0.0 {
        cli.finger_width
    } else {
        finger_joints::DEFAULT_FINGER_WIDTH
    };

    let kerf_values: Vec<f64> = if let Some(ref kerfs) = cli.kerfs {
        let mut v: Vec<f64> = kerfs.iter().map(|&k| (k * 1e6).round() / 1e6).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup();
        v
    } else if cli.kerf_range {
        kerf_sweep(cli.kerf_start, cli.kerf_end, cli.kerf_step)
    } else {
        vec![(cli.kerf * 1e6).round() / 1e6]
    };

    let layout_modes: Vec<String> = if cli.layout == "both" {
        vec!["unfolded".to_string(), "packed".to_string()]
    } else {
        vec![normalize_layout(&cli.layout).to_string()]
    };

    // Step 1: Load STEP and extract planar faces
    println!("Loading STEP file: {}", cli.input);
    let faces = step_loader::load_step(&cli.input, cli.min_area, cli.body)?;

    if faces.is_empty() {
        eprintln!("No qualifying faces found. Try lowering --min-area.");
        std::process::exit(1);
    }

    // Step 2: Find shared edges and classify faces
    println!("Finding shared edges...");
    let shared_edges = face_classifier::find_shared_edges(&faces, 0.1);
    println!("  Found {} shared edges", shared_edges.len());

    println!("Classifying faces...");
    let classification = face_classifier::classify_faces(&faces, &shared_edges);
    let bottom = &classification.bottom;
    let walls = &classification.walls;

    println!(
        "  Bottom plate: face {} (area={:.1} mm²)",
        bottom.face_id, bottom.area
    );
    println!("  Walls: {} faces", walls.len());
    for w in walls {
        println!(
            "    Face {}: area={:.1} mm², normal=({:.2}, {:.2}, {:.2})",
            w.face_id, w.area, w.normal[0], w.normal[1], w.normal[2]
        );
    }
    if !classification.other.is_empty() {
        println!(
            "  Other faces (not connected to bottom): {}",
            classification.other.len()
        );
    }

    // Step 3: Project faces to 2D
    println!("Projecting faces to 2D...");
    let mut relevant_faces: Vec<&PlanarFace> = vec![bottom];
    relevant_faces.extend(walls.iter());

    let mut projections: HashMap<usize, Projection2D> = HashMap::new();
    for face in &relevant_faces {
        let label = if face.face_id == bottom.face_id {
            "bottom".to_string()
        } else {
            let (nx, ny, nz) = (face.normal[0], face.normal[1], face.normal[2]);
            if nx.abs() > ny.abs() && nx.abs() > nz.abs() {
                format!(
                    "wall_x_{}_{}",
                    if nx > 0.0 { "pos" } else { "neg" },
                    face.face_id
                )
            } else if ny.abs() > nz.abs() {
                format!(
                    "wall_y_{}_{}",
                    if ny > 0.0 { "pos" } else { "neg" },
                    face.face_id
                )
            } else {
                format!(
                    "wall_z_{}_{}",
                    if nz > 0.0 { "pos" } else { "neg" },
                    face.face_id
                )
            }
        };

        let proj = projector::project_face(face, &label);
        println!(
            "  {}: {} vertices, {} holes",
            label,
            proj.outer_polygon.len(),
            proj.inner_polygons.len()
        );
        projections.insert(face.face_id, proj);
    }

    // Step 4: Apply finger joints
    let relevant_ids: std::collections::HashSet<usize> = projections.keys().copied().collect();
    let relevant_shared: Vec<SharedEdge> = shared_edges
        .iter()
        .filter(|se| relevant_ids.contains(&se.face_a_id) && relevant_ids.contains(&se.face_b_id))
        .cloned()
        .collect();

    println!(
        "Applying finger joints ({} shared edges)...",
        relevant_shared.len()
    );
    println!("  Thickness: {} mm", cli.thickness);
    println!("  Finger width: {} mm", fw);
    println!(
        "  Kerf values: {} mm",
        kerf_values
            .iter()
            .map(|k| format_mm(*k))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let em = if cli.edge_margin >= 0.0 {
        cli.edge_margin
    } else {
        finger_joints::DEFAULT_EDGE_MARGIN
    };
    let nb = if cli.notch_buffer >= 0.0 {
        cli.notch_buffer
    } else {
        finger_joints::DEFAULT_NOTCH_BUFFER
    };
    let pi = if cli.plateau_inset >= 0.0 {
        cli.plateau_inset
    } else {
        finger_joints::DEFAULT_PLATEAU_INSET
    };
    let mpl = if cli.min_plateau_length >= 0.0 {
        cli.min_plateau_length
    } else {
        finger_joints::DEFAULT_MIN_PLATEAU_LENGTH
    };

    println!("  Edge margin: {} mm", em);
    println!("  Notch buffer: {} mm", nb);
    println!("  Plateau inset: {} mm", pi);
    println!("  Min plateau length: {} mm", mpl);
    println!("  Layouts: {}", layout_modes.join(", "));
    println!("  SVG sheet: {} x {} mm", cli.sheet_width, cli.sheet_height);

    // Step 5: Export
    std::fs::create_dir_all(&cli.output)?;
    let mut written: Vec<String> = Vec::new();
    let mut had_verify_failure = false;
    let file_prefix = Path::new(&cli.input)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());

    for &kerf in &kerf_values {
        let suffix = cut_settings_suffix(cli.thickness, kerf);
        println!("\nProcessing kerf={} mm...", format_mm(kerf));

        let joint_params = FingerJointParams {
            finger_width: fw,
            edge_margin: em,
            notch_buffer: nb,
            plateau_inset: pi,
            min_plateau_length: mpl,
            kerf,
            preserve_outer_dims: true,
        };

        let (modified_polygons, slot_cutouts) = finger_joints::apply_finger_joints(
            &projections,
            &relevant_shared,
            bottom.face_id,
            cli.thickness,
            &joint_params,
            Some(&faces),
        );

        // Report through-slots
        for (&fid, slots) in &slot_cutouts {
            if !slots.is_empty() {
                if let Some(proj) = projections.get(&fid) {
                    println!("  Through-slots on {}: {} slot(s)", proj.label, slots.len());
                }
            }
        }

        // Verification
        if cli.verify || cli.verify_3d {
            let report = verification::verify_joint_mesh(
                &projections,
                &modified_polygons,
                &slot_cutouts,
                &relevant_shared,
                bottom.face_id,
                cli.thickness,
                cli.verify_step,
                cli.verify_tolerance,
                cli.verify_3d,
                cli.verify_interference_tolerance,
                Some(&faces),
            );

            println!(
                "  Verification: {} ({}/{} joints)",
                if report.passed() { "PASS" } else { "FAIL" },
                report.total_joints - report.failed_joints,
                report.total_joints,
            );

            if !report.passed() {
                had_verify_failure = true;
                for r in &report.joints {
                    if !r.passed {
                        println!(
                            "    FAIL {} {}<->{}: {}",
                            r.joint_type, r.face_a_id, r.face_b_id, r.reason
                        );
                    }
                }
                if cli.verify_strict {
                    eprintln!("\nVerification failed and --verify-strict is set.");
                    std::process::exit(2);
                }
            }

            let verify_dir = format!(
                "{}/logs/verification/{}/{}mm/kerf_{}mm",
                cli.output,
                file_prefix,
                format_mm(cli.thickness),
                format_mm(kerf)
            );
            let report_path = format!(
                "{}/{}_{}mm_kerf_{}mm_verify.json",
                verify_dir,
                file_prefix,
                format_mm(cli.thickness),
                format_mm(kerf)
            );
            verification::write_verification_report(&report_path, &report)?;
            println!("  Verification report: {}", report_path);
        }

        // Export
        for layout_mode in &layout_modes {
            let output_dir = settings_output_dir(&cli.output, cli.thickness, kerf, layout_mode);
            std::fs::create_dir_all(&output_dir)?;

            if cli.format.contains(&"dxf".to_string()) {
                println!("Exporting DXF ({})...", layout_mode);
                let dxf_path = if cli.per_face {
                    output_dir.clone()
                } else {
                    format!("{}/{}{}.dxf", output_dir, file_prefix, suffix)
                };
                let files = exporter::export_dxf(
                    &projections,
                    &modified_polygons,
                    &dxf_path,
                    cli.per_face,
                    &slot_cutouts,
                    layout_mode,
                    cli.wall_offset,
                    &relevant_shared,
                    bottom.face_id,
                    Some(&faces),
                    cli.thickness,
                    5.0,
                )?;
                for f in &files {
                    println!("  Written: {}", f);
                }
                written.extend(files);
            }

            if cli.format.contains(&"svg".to_string()) {
                println!("Exporting SVG ({})...", layout_mode);
                let svg_path = format!("{}/{}{}.svg", output_dir, file_prefix, suffix);
                let svg_file = exporter::export_svg(
                    &projections,
                    &modified_polygons,
                    &svg_path,
                    &slot_cutouts,
                    layout_mode,
                    cli.wall_offset,
                    &relevant_shared,
                    bottom.face_id,
                    Some(&faces),
                    cli.thickness,
                    cli.sheet_width,
                    cli.sheet_height,
                    5.0,
                )?;
                println!("  Written: {}", svg_file);
                written.push(svg_file);
            }
        }
    }

    println!(
        "\nDone! {} file(s) written across {} kerf setting(s) and {} layout(s).",
        written.len(),
        kerf_values.len(),
        layout_modes.len()
    );
    if had_verify_failure && !cli.verify_strict {
        println!("Warning: Verification reported failed joints. See reports in output/logs/verification.");
    }

    Ok(())
}
