use anyhow::{bail, Result};
use opencascade_sys::ffi;

use crate::types::{EdgeData, PlanarFace, Vec3};

fn pnt_to_vec3(p: &ffi::gp_Pnt) -> Vec3 {
    [p.X(), p.Y(), p.Z()]
}

fn vec_to_vec3(v: &ffi::gp_Vec) -> Vec3 {
    [v.X(), v.Y(), v.Z()]
}

fn midpoint3(a: Vec3, b: Vec3) -> Vec3 {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0]
}

fn extract_edges_from_wire(
    wire: &ffi::TopoDS_Wire,
    curve_deflection: f64,
) -> Vec<EdgeData> {
    let mut edges = Vec::new();
    let wire_shape = ffi::cast_wire_to_shape(wire);
    let mut explorer = ffi::TopExp_Explorer_ctor(wire_shape, ffi::TopAbs_ShapeEnum::TopAbs_EDGE);

    while explorer.More() {
        let current = ffi::ExplorerCurrentShape(&explorer);
        let edge = ffi::TopoDS_cast_to_edge(&current);

        let mut first_param = 0.0_f64;
        let mut last_param = 0.0_f64;
        let curve_handle = ffi::BRep_Tool_Curve(edge, &mut first_param, &mut last_param);

        if !curve_handle.is_null() {
            let v_first = ffi::TopExp_FirstVertex(edge);
            let v_last = ffi::TopExp_LastVertex(edge);
            let p_first = ffi::BRep_Tool_Pnt(&v_first);
            let p_last = ffi::BRep_Tool_Pnt(&v_last);

            let start = pnt_to_vec3(&p_first);
            let end = pnt_to_vec3(&p_last);
            let edge_len = crate::math_utils::dist3(start, end);

            // Check if edge is straight by sampling the midpoint
            let mid_param = (first_param + last_param) / 2.0;
            let mid_pnt = ffi::HandleGeomCurve_Value(&curve_handle, mid_param);
            let mid = pnt_to_vec3(&mid_pnt);

            let chord_mid = midpoint3(start, end);
            let mid_deviation = crate::math_utils::dist3(mid, chord_mid);

            if mid_deviation < 0.01 {
                edges.push(EdgeData {
                    start,
                    end,
                    midpoint: mid,
                });
            } else {
                // Curved edge: tessellate
                let n_segs = ((edge_len / curve_deflection).ceil() as usize).max(8);
                for i in 0..n_segs {
                    let t0 = first_param
                        + (i as f64 / n_segs as f64) * (last_param - first_param);
                    let t1 = first_param
                        + ((i + 1) as f64 / n_segs as f64) * (last_param - first_param);
                    let t_mid = (t0 + t1) / 2.0;

                    let p0 = ffi::HandleGeomCurve_Value(&curve_handle, t0);
                    let p1 = ffi::HandleGeomCurve_Value(&curve_handle, t1);
                    let pm = ffi::HandleGeomCurve_Value(&curve_handle, t_mid);

                    edges.push(EdgeData {
                        start: pnt_to_vec3(&p0),
                        end: pnt_to_vec3(&p1),
                        midpoint: pnt_to_vec3(&pm),
                    });
                }
            }
        }

        explorer.pin_mut().Next();
    }

    edges
}

/// Load a STEP file and return planar faces above the area threshold.
pub fn load_step(filepath: &str, min_area: f64, _body_index: i32) -> Result<Vec<PlanarFace>> {
    let progress = ffi::Message_ProgressRange_ctor();
    let mut reader = ffi::STEPControl_Reader_ctor();
    let status = ffi::read_step(reader.pin_mut(), filepath.to_string());

    if status != ffi::IFSelect_ReturnStatus::IFSelect_RetDone {
        bail!("Failed to read STEP file: {:?}", status);
    }

    reader.pin_mut().TransferRoots(&progress);
    let shape = ffi::one_shape(&reader);

    let mut faces = Vec::new();
    let mut face_id = 0_usize;

    let mut face_explorer =
        ffi::TopExp_Explorer_ctor(&shape, ffi::TopAbs_ShapeEnum::TopAbs_FACE);

    while face_explorer.More() {
        let current = ffi::ExplorerCurrentShape(&face_explorer);
        let face = ffi::TopoDS_cast_to_face(&current);

        // Get surface properties (area)
        let mut props = ffi::GProp_GProps_ctor();
        ffi::BRepGProp_SurfaceProperties(&current, props.pin_mut());
        let area = props.Mass();

        if area >= min_area {
            // Get face normal using BRepGProp_Face
            let brep_face = ffi::BRepGProp_Face_ctor(face);
            let mut normal_pnt = ffi::new_point(0.0, 0.0, 0.0);
            let mut normal_vec = ffi::new_vec(0.0, 0.0, 1.0);

            // BRepGProp_Face doesn't expose Bounds() in this binding.
            // We'll sample normal at parametric center (0.5, 0.5) as a rough default,
            // then use the BRep surface to get actual UV bounds.
            // Alternatively, sample at (0, 0) which is usually valid for STEP faces.
            //
            // Since Bounds is not available, use a fixed parametric midpoint
            let _surface = ffi::BRep_Tool_Surface(face);

            // Use a fixed parametric midpoint - works for most STEP faces
            let u_mid = 0.5_f64;
            let v_mid = 0.5_f64;

            brep_face.Normal(u_mid, v_mid, normal_pnt.pin_mut(), normal_vec.pin_mut());
            let normal = vec_to_vec3(&normal_vec);

            // Check if face is planar by sampling normals at several points
            let is_planar = check_face_planar(&brep_face, 0.01);

            if is_planar {
                let com = ffi::GProp_GProps_CentreOfMass(&props);
                let center = pnt_to_vec3(&com);

                let mut wire_shapes = Vec::new();
                let face_shape = ffi::cast_face_to_shape(face);
                let mut wire_explorer = ffi::TopExp_Explorer_ctor(
                    face_shape,
                    ffi::TopAbs_ShapeEnum::TopAbs_WIRE,
                );
                while wire_explorer.More() {
                    let wire_current = ffi::ExplorerCurrentShape(&wire_explorer);
                    wire_shapes.push(wire_current);
                    wire_explorer.pin_mut().Next();
                }

                let outer_edges = if !wire_shapes.is_empty() {
                    let wire = ffi::TopoDS_cast_to_wire(&wire_shapes[0]);
                    extract_edges_from_wire(wire, 1.0)
                } else {
                    Vec::new()
                };

                let inner_wires_edges: Vec<Vec<EdgeData>> = wire_shapes[1..]
                    .iter()
                    .map(|ws| {
                        let wire = ffi::TopoDS_cast_to_wire(ws);
                        extract_edges_from_wire(wire, 1.0)
                    })
                    .collect();

                let normal_normalized = crate::math_utils::normalize3(normal);

                faces.push(PlanarFace {
                    face_id,
                    normal: normal_normalized,
                    center,
                    area,
                    outer_wire_edges: outer_edges,
                    inner_wires_edges,
                });
                face_id += 1;
            }
        }

        face_explorer.pin_mut().Next();
    }

    println!(
        "  Found {} planar faces above {} mm² threshold",
        faces.len(),
        min_area
    );
    Ok(faces)
}

/// Check if a face is planar by sampling normals at multiple points.
fn check_face_planar(brep_face: &ffi::BRepGProp_Face, tol: f64) -> bool {
    // Sample at a grid of UV points and check normal consistency.
    // Since we don't have Bounds(), use a set of known-good parametric values.
    let uv_samples = [
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
        (0.5, 0.5),
    ];

    let mut ref_normal: Option<Vec3> = None;

    for &(u, v) in &uv_samples {
        let mut pnt = ffi::new_point(0.0, 0.0, 0.0);
        let mut nvec = ffi::new_vec(0.0, 0.0, 1.0);
        brep_face.Normal(u, v, pnt.pin_mut(), nvec.pin_mut());
        let n = crate::math_utils::normalize3(vec_to_vec3(&nvec));

        if crate::math_utils::norm3(n) < 0.5 {
            continue; // degenerate normal, skip
        }

        match &ref_normal {
            None => ref_normal = Some(n),
            Some(rn) => {
                let dot = crate::math_utils::dot3(*rn, n);
                if (dot - 1.0).abs() > tol {
                    return false;
                }
            }
        }
    }

    true
}
