use crate::types::{Vec2, Vec3};

pub fn dot3(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale3(v: Vec3, s: f64) -> Vec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub fn norm3(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

pub fn normalize3(v: Vec3) -> Vec3 {
    let mag = norm3(v);
    if mag < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / mag, v[1] / mag, v[2] / mag]
}

pub fn dist3(a: Vec3, b: Vec3) -> f64 {
    norm3(sub3(a, b))
}

pub fn points_close3(a: Vec3, b: Vec3, tol: f64) -> bool {
    (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
}

// 2D operations

pub fn dist2(a: Vec2, b: Vec2) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

pub fn lerp2(a: Vec2, b: Vec2, t: f64) -> Vec2 {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

pub fn dot2(a: Vec2, b: Vec2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

pub fn sub2(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] - b[0], a[1] - b[1]]
}

pub fn add2(a: Vec2, b: Vec2) -> Vec2 {
    [a[0] + b[0], a[1] + b[1]]
}

pub fn scale2(v: Vec2, s: f64) -> Vec2 {
    [v[0] * s, v[1] * s]
}

pub fn norm2(v: Vec2) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

pub fn normalize2(v: Vec2) -> Vec2 {
    let mag = norm2(v);
    if mag < 1e-12 {
        return [0.0, 0.0];
    }
    [v[0] / mag, v[1] / mag]
}

pub fn project_point_3d_to_2d(point: Vec3, origin: Vec3, u: Vec3, v: Vec3) -> Vec2 {
    let d = sub3(point, origin);
    [dot3(d, u), dot3(d, v)]
}
