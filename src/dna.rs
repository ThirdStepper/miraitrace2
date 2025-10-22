use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::app::FocusRegion;
use std::sync::{Arc, OnceLock};

/// A polygon with 3-6 points (matching original Evolve's progressive detail)
#[derive(Debug, Serialize, Deserialize)]  // Removed Clone - implemented manually below
pub struct Polygon {
    pub points: Vec<(f32, f32)>,  // 3-6 vertex coordinates
    pub rgba: [f32; 4],            // un-premultiplied, 0..1

    // Cached tiny-skia Path (not serialized, rebuilt on load)
    // Uses OnceLock for lock-free reads after first initialization (Perf C)
    // Path becomes immutable after first computation - vertex changes create new Polygon
    #[serde(skip)]
    pub cached_path: OnceLock<Arc<tiny_skia::Path>>,
}

// Manual Clone implementation that resets cached_path to empty OnceLock
// This prevents stale paths from being copied when Arc::make_mut() clones polygons
impl Clone for Polygon {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            rgba: self.rgba,
            cached_path: OnceLock::new(),  // Always start fresh - never copy cached paths
        }
    }
}

impl Polygon {
    /// Check if any vertex of this polygon intersects the given focus region
    /// (matching Evolve's Poly::hasPointIn method)
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub width: u32,
    pub height: u32,
    // Arc wrapper enables copy-on-write: cloning genome only copies pointers (8 bytes each),
    // not entire polygons. Mutations use Arc::make_mut() to clone only modified polygons.
    // This reduces memory overhead from ~6MB per clone to ~800KB (87% reduction at 100k polys)
    #[serde(with = "arc_vec_serde")]
    pub polys: Vec<Arc<Polygon>>,  // Changed from Vec<Polygon> to enable efficient cloning
}

impl Genome {
    pub fn new_blank(width: u32, height: u32) -> Self {
        profiling::scope!("Genome::new_blank");
        Self { width, height, polys: Vec::new() }
    }

    /// Generate a polygon constrained to a specific focus region (matching Evolve's genPoly with focus)
    pub fn smart_polygon_in_region<R: Rng>(
        &self,
        rng: &mut R,
        target_rgba: &[u8],
        alpha_min: f32,
        alpha_max: f32,
        num_points: usize,
        region: Option<&FocusRegion>,
    ) -> Polygon {
        profiling::scope!("smart_polygon_in_region");
        let w = self.width as f32;
        let h = self.height as f32;

        // Determine bounds based on region
        let (x_min, x_max, y_min, y_max) = if let Some(r) = region {
            (w * r.left, w * r.right, h * r.top, h * r.bottom)
        } else {
            (0.0, w, 0.0, h)
        };

        let width_range = x_max - x_min;
        let height_range = y_max - y_min;

        // Generate random points within the region
        let mut points = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let x = x_min + rng.random::<f32>() * width_range;
            let y = y_min + rng.random::<f32>() * height_range;
            points.push((x, y));
        }

        // Compute center of polygon
        let cx = (points.iter().map(|p| p.0).sum::<f32>() / num_points as f32).clamp(0.0, w - 1.0) as u32;
        let cy = (points.iter().map(|p| p.1).sum::<f32>() / num_points as f32).clamp(0.0, h - 1.0) as u32;

        // Sample 5 points: center, top, bottom, left, right (matching Evolve)
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

// Serde helper module for serializing/deserializing Vec<Arc<T>>
// Arc is transparent for serialization - we serialize the inner value directly
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
