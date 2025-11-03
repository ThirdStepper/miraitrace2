use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::app_types::FocusRegion;
use std::sync::{Arc, OnceLock};

/// a polygon with 3-6 points and color stored as un-premultiplied. also caches a T-S path
#[derive(Debug, Serialize, Deserialize)]
pub struct Polygon {
    pub points: Vec<(f32, f32)>,  // 3-6 vertex coordinates
    pub rgba: [f32; 4],            // un-premultiplied, 0..1

    #[serde(skip)]
    pub cached_path: OnceLock<Arc<tiny_skia::Path>>,
}

// this way stale paths won't be copied if the polygon is cloned.
impl Clone for Polygon {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            rgba: self.rgba,
            cached_path: OnceLock::new(),
        }
    }
}

/// only function is to check if any vertex of this polygon intersects the given focus region
impl Polygon {
    pub fn intersects_region(&self, region: &FocusRegion, width: u32, height: u32) -> bool {
        profiling::scope!("intersects_region");
        let x_min = (width as f32 * region.left) as f32;
        let x_max = (width as f32 * region.right) as f32;
        let y_min = (height as f32 * region.top) as f32;
        let y_max = (height as f32 * region.bottom) as f32;

        self.points.iter().any(|(x, y)| {
            *x >= x_min && *x <= x_max && *y >= y_min && *y <= y_max
        })
    }
}

// arc wrapper enables copy-on-write: cloning genome only copies pointers (8 bytes/each),
// not entire polygons. mutations use Arc::make_mut() to clone only modified polygons.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub width: u32,
    pub height: u32,

    #[serde(with = "arc_vec_serde")]
    pub polys: Vec<Arc<Polygon>>,
}

impl Genome {
    pub fn new_blank(width: u32, height: u32) -> Self {
        profiling::scope!("Genome::new_blank");
        Self { width, height, polys: Vec::new() }
    }

    /// generate a polygon constrained to a specific focus region
    /// With edge-aware seeding: optionally uses edge map to spawn polygons along detected edges
    pub fn smart_polygon_in_region<R: Rng>(
        &self,
        rng: &mut R,
        target_rgba: &[u8],
        alpha_min: f32,
        alpha_max: f32,
        num_points: usize,
        region: Option<&FocusRegion>,
        enforce_simple_convex: bool,
        edge_map: Option<&crate::analysis::EdgeMap>,
        edge_probability: f32,
        edge_vertex_range: f32,
    ) -> Polygon {
        profiling::scope!("smart_polygon_in_region");
        let w = self.width as f32;
        let h = self.height as f32;

        // determine bounds based on region
        let (x_min, x_max, y_min, y_max) = if let Some(r) = region {
            (w * r.left, w * r.right, h * r.top, h * r.bottom)
        } else {
            (0.0, w, 0.0, h)
        };

        let width_range = x_max - x_min;
        let height_range = y_max - y_min;

        // Edge-aware seeding - spawn polygons along detected edges
        let use_edge_seeding = edge_map.is_some() && rng.random::<f32>() < edge_probability;

        let mut points = Vec::with_capacity(num_points);
        if use_edge_seeding {
            // Edge-guided placement
            let emap = edge_map.unwrap();

            // Sample seed point weighted by edge magnitude within the focus region
            // Build cumulative distribution for weighted sampling
            let x_min_u = x_min.max(0.0) as u32;
            let x_max_u = x_max.min(w - 1.0) as u32;
            let y_min_u = y_min.max(0.0) as u32;
            let y_max_u = y_max.min(h - 1.0) as u32;

            let mut total_weight = 0.0f32;
            let mut weights = Vec::new();

            for y in y_min_u..=y_max_u {
                for x in x_min_u..=x_max_u {
                    let mag = emap.sample_magnitude(x, y);
                    total_weight += mag + 0.01; // ε floor to allow non-edge regions
                    weights.push((x, y, total_weight));
                }
            }

            // Sample seed point
            let (seed_x, seed_y, seed_dir) = if total_weight > 0.0 && !weights.is_empty() {
                let threshold = rng.random::<f32>() * total_weight;
                let mut sx = weights[0].0;
                let mut sy = weights[0].1;
                for &(x, y, cumulative) in &weights {
                    if cumulative >= threshold {
                        sx = x;
                        sy = y;
                        break;
                    }
                }
                let dir = emap.sample_direction(sx, sy);
                (sx as f32, sy as f32, dir)
            } else {
                // Fallback: no edges found, use center of region
                ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, 0.0)
            };

            // Place vertices around seed point along edge tangent/normal directions
            // tangent = (cos(dir), sin(dir)), normal = (-sin(dir), cos(dir))
            let tangent = (seed_dir.cos(), seed_dir.sin());
            let normal = (-seed_dir.sin(), seed_dir.cos());

            for i in 0..num_points {
                let angle = (i as f32 / num_points as f32) * 2.0 * std::f32::consts::PI;
                let radius = edge_vertex_range * (0.5 + rng.random::<f32>() * 0.5); // 50-100% of range

                // Offset along tangent/normal based on angle
                let tx = tangent.0 * angle.cos() * radius;
                let ty = tangent.1 * angle.cos() * radius;
                let nx = normal.0 * angle.sin() * radius;
                let ny = normal.1 * angle.sin() * radius;

                let x = (seed_x + tx + nx).clamp(x_min, x_max - 1.0);
                let y = (seed_y + ty + ny).clamp(y_min, y_max - 1.0);
                points.push((x, y));
            }
        } else {
            // Random placement (original logic)
            for _ in 0..num_points {
                let x = x_min + rng.random::<f32>() * width_range;
                let y = y_min + rng.random::<f32>() * height_range;
                points.push((x, y));
            }
        }

        if enforce_simple_convex {
            // try to sanitize (ensures CCW + validates simple + convex)
            if !crate::geom::sanitize_ccw_simple_convex(&mut points) {
                // fallback: sort by angle around centroid to create a proper convex polygon
                let cx = points.iter().map(|p| p.0).sum::<f32>() / points.len() as f32;
                let cy = points.iter().map(|p| p.1).sum::<f32>() / points.len() as f32;
                points.sort_by(|a, b| {
                    let angle_a = (a.1 - cy).atan2(a.0 - cx);
                    let angle_b = (b.1 - cy).atan2(b.0 - cx);
                    angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
                });

                // retry sanitization after angle-sort (should now be valid)
                crate::geom::sanitize_ccw_simple_convex(&mut points);
            }
        }

        // compute center of polygon
        let cx = (points.iter().map(|p| p.0).sum::<f32>() / num_points as f32).clamp(0.0, w - 1.0) as u32;
        let cy = (points.iter().map(|p| p.1).sum::<f32>() / num_points as f32).clamp(0.0, h - 1.0) as u32;

        // sample 5 points: center, top, bottom, left, right 
        let samples = [
            (cx, cy),
            (cx, cy.saturating_sub(5)),                 // top
            (cx, (cy + 5).min(self.height - 1)),        // bottom
            (cx.saturating_sub(5), cy),                 // left
            ((cx + 5).min(self.width - 1), cy),         // right
        ];

        let mut r_sum = 0u32;
        let mut g_sum = 0u32;
        let mut b_sum = 0u32;

        for &(sx, sy) in &samples {
            let idx = ((sy * self.width + sx) * 4) as usize;
            if idx + 2 < target_rgba.len() {
                r_sum += target_rgba[idx] as u32;
                g_sum += target_rgba[idx + 1] as u32;
                b_sum += target_rgba[idx + 2] as u32;
            }
        }

        let rgba = [
            (r_sum as f32 / 5.0 / 255.0),
            (g_sum as f32 / 5.0 / 255.0),
            (b_sum as f32 / 5.0 / 255.0),
            rng.random_range(alpha_min..alpha_max),
        ];

        Polygon {
            points,
            rgba,
            cached_path: OnceLock::new(),
        }
    }

    /// Scale all polygon coordinates by a factor.
    /// Used for progressive multi-resolution evolution.
    /// Factor: 0.25 for 1/4x, 0.5 for 1/2x, 1.0 for 1x (full res)
    pub fn scale_coords(&mut self, factor: f32) {
        for poly_arc in &mut self.polys {
            // Use Arc::make_mut for copy-on-write
            let poly = Arc::make_mut(poly_arc);
            for point in &mut poly.points {
                point.0 *= factor;
                point.1 *= factor;
            }
        }
    }

    /// Create a scaled copy of the genome for multi-resolution rendering.
    /// The new genome will have scaled dimensions and scaled polygon coordinates.
    /// Factor: 0.25 for 1/4x, 0.5 for 1/2x, 1.0 for 1x (full res)
    pub fn create_scaled(&self, factor: f32) -> Self {
        let scaled_width = (self.width as f32 * factor).max(1.0) as u32;
        let scaled_height = (self.height as f32 * factor).max(1.0) as u32;

        let scaled_polys: Vec<Arc<Polygon>> = self.polys.iter().map(|poly_arc| {
            let mut poly = (**poly_arc).clone();
            for point in &mut poly.points {
                point.0 *= factor;
                point.1 *= factor;
            }
            Arc::new(poly)
        }).collect();

        Genome {
            width: scaled_width,
            height: scaled_height,
            polys: scaled_polys,
        }
    }
}

// serde helper module for serializing/deserializing Vec<Arc<T>>. we serialize the inner value directly
mod arc_vec_serde {
    use super::*;
    use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer, SerializeSeq};
    use std::fmt;
    use std::marker::PhantomData;

    pub fn serialize<S, T>(vec: &Vec<Arc<T>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        let mut seq = serializer.serialize_seq(Some(vec.len()))?;
        for item in vec {
            seq.serialize_element(&**item)?;  // Deref Arc to get &T
        }
        seq.end()
    }

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Vec<Arc<T>>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        struct ArcVecVisitor<T>(PhantomData<T>);

        impl<'de, T> Visitor<'de> for ArcVecVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = Vec<Arc<T>>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vec = Vec::new();
                while let Some(item) = seq.next_element::<T>()? {
                    vec.push(Arc::new(item));
                }
                Ok(vec)
            }
        }

        deserializer.deserialize_seq(ArcVecVisitor(PhantomData))
    }
}
