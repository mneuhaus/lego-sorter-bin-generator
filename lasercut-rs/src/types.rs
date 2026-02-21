use serde::{Deserialize, Serialize};

pub type Vec3 = [f64; 3];
pub type Vec2 = [f64; 2];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub start: Vec3,
    pub end: Vec3,
    pub midpoint: Vec3,
}

#[derive(Debug, Clone)]
pub struct PlanarFace {
    pub face_id: usize,
    pub normal: Vec3,
    pub center: Vec3,
    pub area: f64,
    pub outer_wire_edges: Vec<EdgeData>,
    pub inner_wires_edges: Vec<Vec<EdgeData>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedEdge {
    pub face_a_id: usize,
    pub face_b_id: usize,
    pub edge_a: EdgeData,
    pub edge_b: EdgeData,
    pub midpoint: Vec3,
    pub length: f64,
}

#[derive(Debug, Clone)]
pub struct Projection2D {
    pub face_id: usize,
    pub label: String,
    pub outer_polygon: Vec<Vec2>,
    pub inner_polygons: Vec<Vec<Vec2>>,
    pub origin_3d: Vec3,
    pub u_axis: Vec3,
    pub v_axis: Vec3,
    pub normal: Vec3,
    pub outer_edges_2d: Vec<(Vec2, Vec2)>,
    pub edge_map_3d: Vec<EdgeData>,
}

#[derive(Debug, Clone)]
pub struct FaceClassification {
    pub bottom: PlanarFace,
    pub walls: Vec<PlanarFace>,
    pub other: Vec<PlanarFace>,
    pub adjacency: std::collections::HashMap<usize, Vec<(usize, SharedEdge)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointVerification {
    pub joint_id: String,
    pub joint_type: String,
    pub face_a_id: usize,
    pub face_b_id: usize,
    pub samples: usize,
    pub reversed_b: bool,
    pub shift_samples: i32,
    pub mismatch_ratio: f64,
    pub collision_ratio: f64,
    pub double_slot_ratio: f64,
    pub add_coverage_a: f64,
    pub add_coverage_b: f64,
    pub sub_coverage_a: f64,
    pub sub_coverage_b: f64,
    pub passed: bool,
    pub reason: String,
    pub debug_svg: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub total_joints: usize,
    pub failed_joints: usize,
    pub run_interference: bool,
    pub mismatch_tolerance: f64,
    pub interference_tolerance: f64,
    pub joints: Vec<JointVerification>,
}

impl VerificationReport {
    pub fn passed(&self) -> bool {
        self.failed_joints == 0
    }
}
