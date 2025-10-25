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
    pub fn smart_polygon_in_region<R: Rng>(
        &self,
        rng: &mut R,
        target_rgba: &[u8],
        alpha_min: f32,
        alpha_max: f32,
        num_points: usize,
        region: Option<&FocusRegion>,
        enforce_simple_convex: bool,
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

        // generate random points within the region
        let mut points = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let x = x_min + rng.random::<f32>() * width_range;
            let y = y_min + rng.random::<f32>() * height_range;
            points.push((x, y));
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

        // sample 5 points: center, top, bottom, left, right (matching Evolve)
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
