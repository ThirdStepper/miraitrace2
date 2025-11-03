// geometry validation for preventing self-intersecting polygons
//
// this module provides functions to ensure polygons remain:
// - simple (no self-intersections / bow-ties)
// - convex (no inward dents)
// - CCW winding (positive signed area)


/// compute signed area of a polygon using the shoelace formula.
/// returns positive for CCW, negative for CW, zero for degenerate.
pub fn signed_area(pts: &[(f32, f32)]) -> f32 {
    if pts.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    for i in 0..pts.len() {
        let j = (i + 1) % pts.len();
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    area * 0.5
}

/// check if two line segments intersect (proper crossing, not touching).
/// segments: (a, b) and (c, d)
fn segments_intersect(a: (f32, f32), b: (f32, f32), c: (f32, f32), d: (f32, f32)) -> bool {
    // CCW test: returns cross product sign
    fn ccw(p: (f32, f32), q: (f32, f32), r: (f32, f32)) -> f32 {
        (r.1 - p.1) * (q.0 - p.0) - (q.1 - p.1) * (r.0 - p.0)
    }

    let ccw_abc = ccw(a, b, c);
    let ccw_abd = ccw(a, b, d);
    let ccw_cda = ccw(c, d, a);
    let ccw_cdb = ccw(c, d, b);

    // segments intersect if endpoints are on opposite sides of each other
    // use a small epsilon to avoid false positives from numerical precision
    const EPSILON: f32 = 1e-8;
    (ccw_abc * ccw_abd < -EPSILON) && (ccw_cda * ccw_cdb < -EPSILON)
}

/// check if a polygon is simple (no self-intersections).
/// tests all pairs of non-adjacent edges.
pub fn is_simple(pts: &[(f32, f32)]) -> bool {
    let n = pts.len();
    if n < 3 {
        return false;
    }

    // check all pairs of non-adjacent edges
    for i in 0..n {
        let j = (i + 1) % n;
        let edge1 = (pts[i], pts[j]);

        // start k at i+2 to skip adjacent edges
        for k in (i + 2)..n {
            let l = (k + 1) % n;

            // skip if edges share a vertex (adjacent at wraparound)
            if i == l {
                continue;
            }

            let edge2 = (pts[k], pts[l]);
            if segments_intersect(edge1.0, edge1.1, edge2.0, edge2.1) {
                return false;
            }
        }
    }

    true
}

/// check if a polygon is convex and CCW.
/// assumes points are already in order (not necessarily CCW yet).
pub fn is_convex_ccw(pts: &[(f32, f32)]) -> bool {
    let n = pts.len();
    if n < 3 {
        return false;
    }

    // check if area is positive (CCW)
    let area = signed_area(pts);
    if area <= 0.0 {
        return false;
    }

    // check all turns are left turns (positive cross products)
    // allow small negative values for numerical tolerance
    const EPSILON: f32 = -1e-6;

    for i in 0..n {
        let j = (i + 1) % n;
        let k = (j + 1) % n;

        // compute cross product at vertex j
        let dx1 = pts[j].0 - pts[i].0;
        let dy1 = pts[j].1 - pts[i].1;
        let dx2 = pts[k].0 - pts[j].0;
        let dy2 = pts[k].1 - pts[j].1;
        let cross = dx1 * dy2 - dy1 * dx2;

        if cross < EPSILON {
            return false; // right turn or collinear (concave)
        }
    }

    true
}

/// sanitize a polygon to be CCW, simple, and convex.
/// steps:
/// 1. ensure CCW winding (reverse if needed)
/// 2. check if simple and convex (no self-intersections)
///
/// returns true if valid, false if unfixable.
/// modifies `pts` in-place to fix winding if needed.
pub fn sanitize_ccw_simple_convex(pts: &mut [(f32, f32)]) -> bool {
    if pts.len() < 3 {
        return false;
    }

    // step 1
    if signed_area(pts) < 0.0 {
        pts.reverse();
    }

    // step 2
    is_simple(pts) && is_convex_ccw(pts)
}

/// dirty rectangle for tracking which pixels were modified during optimization.
/// used for incremental tile cache updates (only recompute affected tiles).
#[derive(Clone, Copy, Debug)]
pub struct DirtyRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl DirtyRect {
    /// create a new dirty rect from pixel coordinates (inclusive bounds)
    #[inline]
    pub fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Self {
        DirtyRect { x0, y0, x1, y1 }
    }

    /// compute union of two dirty rects (smallest rect containing both)
    #[inline]
    pub fn union(self, other: DirtyRect) -> DirtyRect {
        DirtyRect {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_area_ccw() {
        // Square with CCW winding
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(signed_area(&pts) > 0.0);
    }

    #[test]
    fn test_signed_area_cw() {
        // Square with CW winding
        let pts = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        assert!(signed_area(&pts) < 0.0);
    }

    #[test]
    fn test_bow_tie_rejected() {
        // Bow-tie (self-intersecting)
        let pts = vec![(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(!is_simple(&pts));
    }

    #[test]
    fn test_convex_square_accepted() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(is_simple(&pts));
        assert!(is_convex_ccw(&pts));
    }

    #[test]
    fn test_concave_quad_rejected() {
        // Concave quad: 3 corners of a square, plus one point pushed inside
        // Makes a "pac-man" mouth shape
        //   (0,1)---(0.3,0.5)
        //     |       /
        //     |     /
        //   (0,0)---(1,0)
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.3, 0.5), (0.0, 1.0)];
        assert!(!is_convex_ccw(&pts));
    }

    #[test]
    fn test_sanitize_fixes_winding() {
        // CW square
        let mut pts = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        assert!(sanitize_ccw_simple_convex(&mut pts));
        // Should be CCW now
        assert!(signed_area(&pts) > 0.0);
    }
}

/// Compute polygon area using Shoelace formula
/// Returns absolute area in square pixels
#[inline]
pub fn polygon_area(points: &[(f32, f32)]) -> f32 {
    signed_area(points).abs()
}
