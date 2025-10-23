use tiny_skia as sk;
use crate::dna::{Genome, Polygon};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

// Global anti-aliasing setting (can be changed from settings UI)
static POLYGON_ANTIALIASING: AtomicBool = AtomicBool::new(true);

/// Update the polygon anti-aliasing setting (called from settings UI)
pub fn set_polygon_antialiasing(enabled: bool) {
    POLYGON_ANTIALIASING.store(enabled, Ordering::Relaxed);
}

// Scratch pixmap reused across calls to avoid allocations.
// Lives at module scope so it's shared within this module.
thread_local! {
    static SCRATCH_PIX: std::cell::RefCell<Option<sk::Pixmap>> =
        std::cell::RefCell::new(None);
    // Cached output buffer to avoid repeated Vec allocations in hot path
    static SCRATCH_VEC: std::cell::RefCell<Vec<u8>> =
        std::cell::RefCell::new(Vec::new());
}

pub struct CpuRenderer;

impl CpuRenderer {
    /// Full-frame render to premultiplied RGBA (tiny-skia's native format).
    /// This is zero-copy - just returns the pixmap's internal buffer.
    /// egui supports premultiplied format natively, so no conversion needed.
    pub fn render_rgba_premul(genome: &Genome) -> Vec<u8> {
        profiling::scope!("render_rgba_premul");
        let w = genome.width;
        let h = genome.height;
        let mut pix = sk::Pixmap::new(w, h).expect("pixmap");
        // White background (classic Evolve-style)
        pix.fill(sk::Color::from_rgba(1.0, 1.0, 1.0, 1.0).unwrap());

        for poly in &genome.polys {
            draw_polygon(&mut pix, poly, sk::Transform::identity());
        }
        // Return premultiplied data directly (zero-copy, no conversion)
        pix.data().to_vec()
    }

    // === New: render only up-to index, returning PREMULTIPLIED RGBA once ===
    pub fn render_up_to_poly_premul(genome: &Genome, up_to_index: usize) -> Vec<u8> {
        profiling::scope!("render_up_to_poly_premul");
    
        let w = genome.width;
        let h = genome.height;
    
        let mut pix = sk::Pixmap::new(w, h).expect("pixmap");
        // White background (classic Evolve)
        pix.fill(sk::Color::from_rgba(1.0, 1.0, 1.0, 1.0).unwrap());
    
        for i in 0..up_to_index.min(genome.polys.len()) {
            draw_polygon(&mut pix, &genome.polys[i], sk::Transform::identity());
        }
    
        // tiny-skia stores PREMULTIPLIED bytes internally:
        pix.data().to_vec()
    }

    // === FAST PREMULT version: render from index onto base, return PREMULT ===
    // Returns premultiplied directly for use in optimization hot path
    // Also reuses output Vec to eliminate allocation overhead
    pub fn render_from_poly_on_base_premul_fast(
        genome: &Genome,
        from_index: usize,
        base_premul: &[u8],
    ) -> Vec<u8> {
        profiling::scope!("render_from_poly_on_base_premul_fast");

        let w = genome.width;
        let h = genome.height;

        SCRATCH_VEC.with(|vec_cell| {
            SCRATCH_PIX.with(|pix_cell| {
                let need_new = match pix_cell.borrow().as_ref() {
                    Some(pm) => pm.width() != w || pm.height() != h,
                    None => true,
                };
                if need_new {
                    *pix_cell.borrow_mut() = Some(sk::Pixmap::new(w, h).expect("scratch pixmap"));
                }

                let mut pix_borrow = pix_cell.borrow_mut();
                let pix = pix_borrow.as_mut().expect("scratch pixmap present");

                let dst = pix.data_mut();
                debug_assert_eq!(dst.len(), base_premul.len());
                dst.copy_from_slice(base_premul);

                for i in from_index..genome.polys.len() {
                    draw_polygon(pix, &genome.polys[i], sk::Transform::identity());
                }

                // Move out the Vec with zero-copy instead of cloning (Perf A)
                let mut vec_borrow = vec_cell.borrow_mut();
                vec_borrow.clear();
                vec_borrow.extend_from_slice(pix.data());
                std::mem::take(&mut *vec_borrow)
            })
        })
    }


}

fn draw_polygon(pix: &mut sk::Pixmap, poly: &Polygon, transform: sk::Transform) {
    profiling::scope!("draw_polygon");
    if poly.points.is_empty() {
        return;
    }

    // Quick reject: integer bbox fully outside the pixmap
    let (w, h) = (pix.width(), pix.height());
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for &(x, y) in &poly.points {
        if x < min_x { min_x = x; }
        if y < min_y { min_y = y; }
        if x > max_x { max_x = x; }
        if y > max_y { max_y = y; }
    }
    if max_x < 0.0 || max_y < 0.0 || min_x >= w as f32 || min_y >= h as f32 {
        return; // fully off-screen: skip tiny-skia work
    }

    // Use cached path with lock-free reads (OnceLock - Perf C)
    // get_or_init populates cache on first call, subsequent calls are lock-free
    let path = poly.cached_path.get_or_init(|| {
        // Build path once, never rebuild (vertices are immutable after creation)
        let mut pb = sk::PathBuilder::new();
        pb.move_to(poly.points[0].0, poly.points[0].1);
        for i in 1..poly.points.len() {
            pb.line_to(poly.points[i].0, poly.points[i].1);
        }
        pb.close();
        Arc::new(pb.finish().expect("path build"))
    });

    let color = sk::Color::from_rgba(poly.rgba[0], poly.rgba[1], poly.rgba[2], poly.rgba[3]).unwrap();
    let mut paint = sk::Paint::default();
    paint.anti_alias = POLYGON_ANTIALIASING.load(Ordering::Relaxed);
    paint.shader = sk::Shader::SolidColor(color);

    let fill_rule = sk::FillRule::Winding;
    pix.fill_path(path, &paint, fill_rule, transform, None);
}

/// Premultiply RGBA - optimized scalar implementation (compiler will auto-vectorize)
#[inline(always)]
pub fn premultiply(p: &[u8]) -> Vec<u8> {
    profiling::scope!("premultiply");

    let mut out = vec![0u8; p.len()];
    let n = p.len();
    let mut i = 0usize;

    while i < n {
        let a = p[i + 3] as u16;
        // (x * a + 127) / 255 is a fast rounded divide-by-255
        out[i]     = ((p[i]     as u16 * a + 127) / 255) as u8;
        out[i + 1] = ((p[i + 1] as u16 * a + 127) / 255) as u8;
        out[i + 2] = ((p[i + 2] as u16 * a + 127) / 255) as u8;
        out[i + 3] = a as u8;
        i += 4;
    }

    out
}
