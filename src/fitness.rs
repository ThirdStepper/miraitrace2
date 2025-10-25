/// Sum of Absolute Differences (SAD) / Manhattan distance on RGBA (all 4 channels).
/// note: alpha is included because we work with premultiplied alpha, where alpha affects blending.
use rayon::prelude::*;
use crate::dna::Polygon;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated SAD computation using x86_64 PSADBW instruction.
/// processes 16 bytes at a time using specialized Sum of Absolute Differences instruction.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn sad_simd_chunk(target: &[u8], current: &[u8]) -> u64 {
    debug_assert!(target.len() == current.len());
    debug_assert!(target.len() % 16 == 0);

    let mut sum = _mm_setzero_si128();
    let chunks = target.len() / 16;

    for i in 0..chunks {
        let offset = i * 16;

        // load 16 bytes (full 128-bit register) from each buffer
        let t_bytes = _mm_loadu_si128(target.as_ptr().add(offset) as *const __m128i);
        let c_bytes = _mm_loadu_si128(current.as_ptr().add(offset) as *const __m128i);

        // PSADBW: Sum of Absolute Differences (16 bytes → two u64 sums, one per 64-bit lane)
        let sad = _mm_sad_epu8(t_bytes, c_bytes);
        sum = _mm_add_epi64(sum, sad);
    }

    // extract and sum both 64-bit lanes (each contains sum of 8 bytes)
    let low = _mm_cvtsi128_si64(sum) as u64;
    let high = _mm_extract_epi64(sum, 1) as u64;
    low + high
}

/// parallel SIMD-accelerated SAD using Rayon + x86_64 intrinsics.
/// matches the original Evolve's multi-core SIMD implementation.
/// early-exit optimization: If best_so_far is provided and accumulated SAD exceeds it,
/// returns u64::MAX immediately to signal "definitely worse than current best".
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn sad_rgb_parallel(target_rgba: &[u8], current_rgba: &[u8], best_so_far: Option<u64>) -> f64 {
    debug_assert_eq!(target_rgba.len(), current_rgba.len());
    debug_assert_eq!(target_rgba.len() % 4, 0);

    let len = target_rgba.len();

    // process in chunks aligned to 16-byte SIMD boundaries
    // round down to nearest multiple of 16
    let simd_len = (len / 16) * 16;

    // parallel SIMD processing with minimum chunk size to reduce task overhead
    // minimum chunk: 256 KB = 262144 bytes (helps small/medium images)
    const MIN_CHUNK_BYTES: usize = 256 * 1024;
    let num_cores = rayon::current_num_threads();
    let chunk_size = if simd_len > 0 {
        let ideal_chunk = (simd_len / num_cores / 16) * 16; // align to 16-byte boundaries
        ideal_chunk.max(MIN_CHUNK_BYTES)
    } else {
        0
    };

    let simd_sum: u64 = if chunk_size > 0 && chunk_size <= simd_len {
        // early-exit path: if we have a best_so_far, check periodically
        if let Some(threshold) = best_so_far {
            let mut acc = 0u64;
            for (t_chunk, c_chunk) in target_rgba[..simd_len]
                .chunks(chunk_size)
                .zip(current_rgba[..simd_len].chunks(chunk_size))
            {
                let chunk_sad = unsafe { sad_simd_chunk(t_chunk, c_chunk) };
                acc += chunk_sad;
                // check after each chunk (every 256KB typically)
                if acc >= threshold {
                    return u64::MAX as f64; // early exit - definitely worse
                }
            }
            acc
        } else {
            // no early-exit: parallel sum as before
            target_rgba[..simd_len]
                .par_chunks(chunk_size)
                .zip(current_rgba[..simd_len].par_chunks(chunk_size))
                .map(|(t_chunk, c_chunk)| unsafe {
                    sad_simd_chunk(t_chunk, c_chunk)
                })
                .sum()
        }
    } else {
        0
    };

    // handle remainder bytes with scalar code (should be < 16 bytes = 0-3 pixels)
    // note: include alpha to match SIMD behavior (PSADBW processes all 16 bytes including alpha)
    let remainder_pixels = (target_rgba.len() - simd_len) / 4;
    let scalar_sum: u64 = (0..remainder_pixels)
        .map(|i| {
            let idx = simd_len + i * 4;
            let r_diff = (target_rgba[idx] as i32 - current_rgba[idx] as i32).abs() as u64;
            let g_diff = (target_rgba[idx + 1] as i32 - current_rgba[idx + 1] as i32).abs() as u64;
            let b_diff = (target_rgba[idx + 2] as i32 - current_rgba[idx + 2] as i32).abs() as u64;
            let a_diff = (target_rgba[idx + 3] as i32 - current_rgba[idx + 3] as i32).abs() as u64;
            r_diff + g_diff + b_diff + a_diff
        })
        .sum();

    let total = simd_sum + scalar_sum;

    // final check against threshold if provided
    if let Some(threshold) = best_so_far {
        if total >= threshold {
            return u64::MAX as f64;
        }
    }

    total as f64
}

/// fallback non-SIMD version for non-x86_64 platforms
/// note: includes alpha channel to match x86_64 SIMD behavior
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn sad_rgb_parallel(target_rgba: &[u8], current_rgba: &[u8], best_so_far: Option<u64>) -> f64 {
    debug_assert_eq!(target_rgba.len(), current_rgba.len());
    debug_assert_eq!(target_rgba.len() % 4, 0);

    let pixels = target_rgba.len() / 4;

    // early-exit path: sequential processing with periodic checks
    if let Some(threshold) = best_so_far {
        let mut acc = 0u64;
        const CHECK_INTERVAL: usize = 1024; // check every 1024 bytes = 256 pixels

        for i in 0..pixels {
            unsafe {
                // safety: we asserted 4-byte stride above
                let t0 = *target_rgba.get_unchecked(i * 4);
                let t1 = *target_rgba.get_unchecked(i * 4 + 1);
                let t2 = *target_rgba.get_unchecked(i * 4 + 2);
                let t3 = *target_rgba.get_unchecked(i * 4 + 3);

                let s0 = *current_rgba.get_unchecked(i * 4);
                let s1 = *current_rgba.get_unchecked(i * 4 + 1);
                let s2 = *current_rgba.get_unchecked(i * 4 + 2);
                let s3 = *current_rgba.get_unchecked(i * 4 + 3);

                let d0 = (t0 as i32 - s0 as i32).abs() as u64;
                let d1 = (t1 as i32 - s1 as i32).abs() as u64;
                let d2 = (t2 as i32 - s2 as i32).abs() as u64;
                let d3 = (t3 as i32 - s3 as i32).abs() as u64;
                acc += d0 + d1 + d2 + d3;
            }

            // check every CHECK_INTERVAL pixels (power of 2 for efficient masking)
            if (i & (CHECK_INTERVAL - 1)) == 0 && acc >= threshold {
                return u64::MAX as f64; // early exit - definitely worse
            }
        }
        return acc as f64;
    }

    // no early-exit: parallel processing as before
    let min_chunk = 64 * 1024; // pixels per Rayon "unit"
    let total: u64 = (0..pixels)
        .into_par_iter()
        .with_min_len(min_chunk)
        .map(|i| unsafe {
            // safety: we asserted 4-byte stride above
            let t0 = *target_rgba.get_unchecked(i * 4);
            let t1 = *target_rgba.get_unchecked(i * 4 + 1);
            let t2 = *target_rgba.get_unchecked(i * 4 + 2);
            let t3 = *target_rgba.get_unchecked(i * 4 + 3);

            let s0 = *current_rgba.get_unchecked(i * 4);
            let s1 = *current_rgba.get_unchecked(i * 4 + 1);
            let s2 = *current_rgba.get_unchecked(i * 4 + 2);
            let s3 = *current_rgba.get_unchecked(i * 4 + 3);

            let d0 = (t0 as i32 - s0 as i32).abs() as u64;
            let d1 = (t1 as i32 - s1 as i32).abs() as u64;
            let d2 = (t2 as i32 - s2 as i32).abs() as u64;
            let d3 = (t3 as i32 - s3 as i32).abs() as u64;
            d0 + d1 + d2 + d3
        })
        .sum();

    total as f64
}

/// ---- luminance-weighted fitness for perceptual bright-region emphasis ----

/// Approximate BT.709 luma in 0..255 using integer math (no floating point).
/// Y ≈ 0.2126*R + 0.7152*G + 0.0722*B  →  (54*R + 183*G + 19*B) >> 8
/// This gives perceptually-weighted luminance suitable for brightness calculations.
#[inline]
fn luma_709_u8(r: u8, g: u8, b: u8) -> u8 {
    let y = 54u32 * (r as u32) + 183u32 * (g as u32) + 19u32 * (b as u32);
    ((y >> 8).min(255)) as u8
}

/// Precompute per-pixel Q8.8 fixed-point weights from target *unpremultiplied* RGBA.
///
/// **Corrected formula:** w = 1.0 + k * (Y/255)
/// where k_q8 is Q8.8 fixed-point (e.g., k=48 in Q8.8 means 48/256 ≈ 0.1875)
///
/// - k_q8: perceptual emphasis parameter in Q8.8 format (0 = disabled, 48 ≈ 0.1875 is balanced)
/// - Y: BT.709 luma (0-255) computed from unpremultiplied RGB (accurate reflectance)
/// - scale_by_alpha: if true, multiply weight by (alpha/255) to further de-emphasize transparent pixels
///   (default: false, since premultiplied RGB already encodes coverage)
///
/// Returns: Vec of Q8.8 weights (one per pixel), where 256 = 1.0
///
/// Examples (scale_by_alpha=false):
/// - k=0: all weights = 256 (1.0×) → equivalent to unweighted SAD
/// - k=48, white pixel (255,255,255): w = 256 + 48 = 304 (1.1875×) → 19% extra weight
/// - k=48, mid-gray (128,128,128): w = 256 + 24 = 280 (1.09375×) → 9% extra weight
/// - k=48, black (0,0,0): w = 256 (1.0×) → no extra weight
/// - k=96, white pixel: w = 256 + 96 = 352 (1.375×) → 38% extra weight
pub fn precompute_luma_weights_q8(
    target_unpremul_rgba: &[u8],
    k_q8: u16,
    scale_by_alpha: bool,
) -> Vec<u16> {
    profiling::scope!("precompute_luma_weights_q8");
    assert!(target_unpremul_rgba.len() % 4 == 0, "RGBA buffer must have length divisible by 4");

    let mut out = Vec::with_capacity(target_unpremul_rgba.len() / 4);
    for px in target_unpremul_rgba.chunks_exact(4) {
        let (r, g, b, a) = (px[0], px[1], px[2], px[3]);
        let y = luma_709_u8(r, g, b) as u16;    // 0..255

        // CORRECTED Q8.8 math: term = (k_q8 * y) / 255, still in Q8.8
        // +127 for rounding (equivalent to adding 0.5 before truncating)
        let term_q8 = ((k_q8 as u32 * y as u32) + 127) / 255;
        let mut w_q8 = 256u32 + term_q8;        // 1.0 in Q8.8 + term

        // Optional: scale by alpha (off by default to avoid double-attenuation with premul RGB)
        if scale_by_alpha {
            // Multiply by (alpha/255) while remaining in Q8.8
            w_q8 = (w_q8 * (a as u32) + 127) / 255;
        }

        // Ensure weight is at least 1 to avoid zero-weight pixels (safety)
        out.push((w_q8 as u16).max(1));
    }
    out
}

/// Downsample Q8.8 weight buffer by 2× using box filter (matches image pyramid downsampling).
/// This creates a weight pyramid level for use with coarse-to-fine fitness evaluation.
///
/// The box filter averages 4 weights (2×2 neighborhood) to get the downsampled weight,
/// ensuring weight magnitudes remain consistent across pyramid levels.
pub fn downsample_weights_q8_box2(src: &[u16], src_w: u32, src_h: u32) -> Vec<u16> {
    profiling::scope!("downsample_weights_q8_box2");

    let dst_w = (src_w + 1) / 2;
    let dst_h = (src_h + 1) / 2;
    let mut out = vec![0u16; (dst_w * dst_h) as usize];

    for y in 0..dst_h {
        let sy0 = (y * 2).min(src_h - 1);
        let sy1 = (sy0 + 1).min(src_h - 1);
        for x in 0..dst_w {
            let sx0 = (x * 2).min(src_w - 1);
            let sx1 = (sx0 + 1).min(src_w - 1);

            let idx = |xx: u32, yy: u32| (yy * src_w + xx) as usize;
            let i00 = idx(sx0, sy0);
            let i10 = idx(sx1, sy0);
            let i01 = idx(sx0, sy1);
            let i11 = idx(sx1, sy1);

            // Average 4 weights (box filter)
            let sum = src[i00] as u32 + src[i10] as u32 + src[i01] as u32 + src[i11] as u32;
            out[(y * dst_w + x) as usize] = ((sum + 2) >> 2) as u16; // rounded average
        }
    }
    out
}

/// Weighted RGB SAD over full image using Q8.8 fixed-point weights.
///
/// Computes: sum over all pixels of (|R_target - R_current| + |G_target - G_current| + |B_target - B_current|) * weight
///
/// Note: Alpha channel difference is NOT included because coverage is already handled by the weight
/// (weights are pre-scaled by alpha during precomputation).
///
/// Returns: Weighted SAD as f64 (Q8.8 converted back to integer by >> 8)
pub fn sad_rgb_weighted_q8(target_rgba: &[u8], current_rgba: &[u8], weights_q8: &[u16]) -> f64 {
    profiling::scope!("sad_rgb_weighted_q8");
    debug_assert_eq!(target_rgba.len(), current_rgba.len());
    debug_assert_eq!(weights_q8.len() * 4, target_rgba.len(), "One weight per pixel required");

    let mut acc: u64 = 0;
    for (i, t) in target_rgba.chunks_exact(4).enumerate() {
        let c = &current_rgba[i * 4..i * 4 + 4];
        let w = weights_q8[i] as u64; // Q8.8 weight

        // RGB differences only (alpha handled by weight pre-scaling)
        let dr = (t[0] as i32 - c[0] as i32).abs() as u64;
        let dg = (t[1] as i32 - c[1] as i32).abs() as u64;
        let db = (t[2] as i32 - c[2] as i32).abs() as u64;
        let rgb_diff = dr + dg + db;

        acc = acc.wrapping_add(rgb_diff * w);
    }

    // Convert from Q8.8 back to integer
    (acc >> 8) as f64
}

/// Weighted RGB SAD over a rectangular tile using Q8.8 fixed-point weights.
/// Used for tiled fitness caching with perceptual weighting.
///
/// Parameters match sad_rgb_rect but with added weights_q8 parameter.
/// stride: width of full image in pixels
/// x_min, y_min, x_max, y_max: tile bounds (inclusive)
#[inline]
pub fn sad_rgb_weighted_q8_rect(
    target: &[u8],
    current: &[u8],
    weights_q8: &[u16],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
) -> f64 {
    profiling::scope!("sad_rgb_weighted_q8_rect");

    let mut acc: u64 = 0;
    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let pixel_idx = (y * stride + x) as usize;
            let byte_idx = pixel_idx * 4;
            let w = weights_q8[pixel_idx] as u64;

            // RGB differences only
            let dr = (target[byte_idx] as i32 - current[byte_idx] as i32).abs() as u64;
            let dg = (target[byte_idx + 1] as i32 - current[byte_idx + 1] as i32).abs() as u64;
            let db = (target[byte_idx + 2] as i32 - current[byte_idx + 2] as i32).abs() as u64;

            acc = acc.wrapping_add((dr + dg + db) * w);
        }
    }

    (acc >> 8) as f64
}

/// ---- gaussian pyramid (RGBA premul) for coarse-to-fine fitness evaluation ----

/// multi-resolution pyramid with 3 levels: 1/4x, 1/2x, 1x
pub struct GaussianPyramid {
    /// level 0 = 1/4x, level 1 = 1/2x, level 2 = 1x (full resolution)
    pub levels: Vec<Vec<u8>>,
    pub widths: Vec<u32>,
    pub heights: Vec<u32>,
}

/// box filter downsample by 2x (simple and fast, good enough for fitness approximation)
#[inline]
fn box_down_2x_rgba(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let dst_w = (w + 1) / 2;
    let dst_h = (h + 1) / 2;
    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];

    for y in 0..dst_h {
        let sy0 = (y * 2).min(h - 1);
        let sy1 = (sy0 + 1).min(h - 1);
        for x in 0..dst_w {
            let sx0 = (x * 2).min(w - 1);
            let sx1 = (sx0 + 1).min(w - 1);

            let idx = |xx: u32, yy: u32| ((yy * w + xx) * 4) as usize;
            let i00 = idx(sx0, sy0);
            let i10 = idx(sx1, sy0);
            let i01 = idx(sx0, sy1);
            let i11 = idx(sx1, sy1);
            let o = ((y * dst_w + x) * 4) as usize;

            // simple box filter: average 4 samples
            for c in 0..4 {
                let s = src[i00 + c] as u32
                    + src[i10 + c] as u32
                    + src[i01 + c] as u32
                    + src[i11 + c] as u32;
                out[o + c] = (s >> 2) as u8;
            }
        }
    }
    (out, dst_w, dst_h)
}

/// build 3-level Gaussian pyramid from premultiplied RGBA image
/// returns pyramid with levels: 0 = 1/4x, 1 = 1/2x, 2 = 1x (original)
pub fn build_pyramid_rgba(premul_rgba: &[u8], w: u32, h: u32) -> GaussianPyramid {
    profiling::scope!("build_pyramid_rgba");

    // level 2: full resolution (1x)
    let l2 = premul_rgba.to_vec();

    // level 1: half resolution (1/2x)
    let (l1, w1, h1) = box_down_2x_rgba(&l2, w, h);

    // level 0: quarter resolution (1/4x)
    let (l0, w0, h0) = box_down_2x_rgba(&l1, w1, h1);

    GaussianPyramid {
        levels: vec![l0, l1, l2],
        widths: vec![w0, w1, w],
        heights: vec![h0, h1, h],
    }
}

/// SAD over a rect at a specific pyramid level
/// `rect` coords (x_min, y_min, x_max, y_max) are in FULL-RES coordinates
/// `scale_div` is the downsampling factor (1, 2, or 4)
#[inline]
pub fn sad_rgb_rect_pyr_level(
    target_lvl: &[u8],
    lvl_w: u32,
    lvl_h: u32,
    scale_div: u32,
    current_full: &[u8],
    full_w: u32,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    best_so_far: Option<u64>,
) -> f64 {
    // map full-res coords to this pyramid level
    let sx = |v: u32| (v / scale_div).min(lvl_w - 1);
    let sy = |v: u32| (v / scale_div).min(lvl_h - 1);

    let lx0 = sx(x_min);
    let ly0 = sy(y_min);
    let lx1 = sx(x_max);
    let ly1 = sy(y_max);

    let mut acc: u64 = 0;

    // sample current image at downsampled locations (nearest neighbor)
    for ly in ly0..=ly1 {
        let y = (ly * scale_div).min(y_max);
        let tr = (ly * lvl_w * 4) as usize;

        for lx in lx0..=lx1 {
            let x = (lx * scale_div).min(x_max);
            let ti = tr + (lx * 4) as usize;
            let ci = ((y * full_w + x) * 4) as usize;

            // SAD on all 4 channels (RGBA premul)
            acc += (target_lvl[ti] as i32 - current_full[ci] as i32).abs() as u64;
            acc += (target_lvl[ti + 1] as i32 - current_full[ci + 1] as i32).abs() as u64;
            acc += (target_lvl[ti + 2] as i32 - current_full[ci + 2] as i32).abs() as u64;
            acc += (target_lvl[ti + 3] as i32 - current_full[ci + 3] as i32).abs() as u64;
        }

        // early abort if exceeding threshold
        if let Some(t) = best_so_far {
            if acc >= t {
                return u64::MAX as f64;
            }
        }
    }

    acc as f64
}

/// coarse-to-fine SAD for a rect: test 1/4x → 1/2x → 1x with early abort
/// compares candidate rect vs current rect at each pyramid level.
/// if candidate is not better than current at any level, immediately returns f64::INFINITY.
pub fn sad_rgb_rect_pyramid(
    pyr: &GaussianPyramid,
    current_old_full: &[u8],  // current render before mutation
    current_new_full: &[u8],  
    full_w: u32,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
) -> f64 {
    profiling::scope!("sad_rgb_rect_pyramid");

    // level 0: 1/4x (coarsest, fastest)
    let sad_new_quarter = sad_rgb_rect_pyr_level(
        &pyr.levels[0],
        pyr.widths[0],
        pyr.heights[0],
        4,
        current_new_full,
        full_w,
        x_min,
        y_min,
        x_max,
        y_max,
        None,
    );
    let sad_old_quarter = sad_rgb_rect_pyr_level(
        &pyr.levels[0],
        pyr.widths[0],
        pyr.heights[0],
        4,
        current_old_full,
        full_w,
        x_min,
        y_min,
        x_max,
        y_max,
        None,
    );
    if sad_new_quarter >= sad_old_quarter {
        return f64::INFINITY; // Early abort at 1/4x - candidate not better
    }

    // level 1: 1/2x (medium detail)
    let sad_new_half = sad_rgb_rect_pyr_level(
        &pyr.levels[1],
        pyr.widths[1],
        pyr.heights[1],
        2,
        current_new_full,
        full_w,
        x_min,
        y_min,
        x_max,
        y_max,
        None,
    );
    let sad_old_half = sad_rgb_rect_pyr_level(
        &pyr.levels[1],
        pyr.widths[1],
        pyr.heights[1],
        2,
        current_old_full,
        full_w,
        x_min,
        y_min,
        x_max,
        y_max,
        None,
    );
    if sad_new_half >= sad_old_half {
        return f64::INFINITY; // Early abort at 1/2x - candidate not better
    }

    // level 2: 1x (full resolution, exact)
    sad_rgb_rect(
        &pyr.levels[2],
        current_new_full,
        x_min,
        y_min,
        x_max,
        y_max,
        full_w,
        None,
    )
}

/// compute axis-aligned bounding box of a polygon with anti-aliasing padding.
/// returns (x_min, y_min, x_max, y_max) in pixel coordinates, clamped to image bounds.
/// AA padding extends the bbox by ~2 pixels to account for anti-aliasing.
pub fn poly_bounds_aa(poly: &Polygon, width: u32, height: u32) -> (u32, u32, u32, u32) {
    profiling::scope!("poly_bounds_aa");

    if poly.points.is_empty() {
        return (0, 0, 0, 0);
    }

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

    // add AA padding (2 pixels) and clamp to image bounds
    const AA_PAD: f32 = 2.0;
    let x_min = (min_x - AA_PAD).max(0.0) as u32;
    let y_min = (min_y - AA_PAD).max(0.0) as u32;
    let x_max = (max_x + AA_PAD).min(width as f32 - 1.0).ceil() as u32;
    let y_max = (max_y + AA_PAD).min(height as f32 - 1.0).ceil() as u32;

    (x_min, y_min, x_max, y_max)
}

/// SIMD (AVX2) SAD computation for rectangular region
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad_rgb_rect_avx2(
    target: &[u8],
    current: &[u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
    best_so_far: Option<u64>,
) -> f64 {
    let mut sum: u64 = 0;
    let row_width = (x_max - x_min + 1) as usize;
    let row_bytes = row_width * 4; // 4 bytes per pixel (RGBA)
    const CHECK_INTERVAL: u32 = 8; // Check every 8 rows

    // process each row
    for y in y_min..=y_max {
        let row_start = ((y * stride + x_min) * 4) as usize;

        // process row with AVX2 (32 bytes = 8 pixels at a time)
        let mut row_sum = _mm256_setzero_si256();
        let simd_pixels = (row_width / 8) * 8;
        let simd_bytes = simd_pixels * 4;

        let mut i = 0;
        while i + 32 <= simd_bytes {
            let idx = row_start + i;

            // load 32 bytes (8 pixels) from target and current
            let target_vec = _mm256_loadu_si256(target.as_ptr().add(idx) as *const __m256i);
            let current_vec = _mm256_loadu_si256(current.as_ptr().add(idx) as *const __m256i);

            // compute absolute difference with saturation
            // sad_epu8 computes horizontal sums in 64-bit lanes
            let sad = _mm256_sad_epu8(target_vec, current_vec);
            row_sum = _mm256_add_epi64(row_sum, sad);
            i += 32;
        }

        // extract and sum the four 64-bit values from the AVX2 register
        let mut temp: [u64; 4] = [0; 4];
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, row_sum);
        sum += temp[0] + temp[1] + temp[2] + temp[3];

        // handle remaining pixels in row with scalar code
        // start from where SIMD left off (up to 31 bytes may remain)
        let mut j = i;
        while j < row_bytes {
            let idx = row_start + j;
            // bounds safety check for scalar path
            if idx >= target.len() || idx >= current.len() {
                break;
            }
            let diff = (target[idx] as i32 - current[idx] as i32).abs() as u64;
            sum += diff;
            j += 1;
        }

        // early-exit check every CHECK_INTERVAL rows
        if let Some(threshold) = best_so_far {
            if (y - y_min) % CHECK_INTERVAL == 0 && sum >= threshold {
                return u64::MAX as f64;
            }
        }
    }

    sum as f64
}

/// scalar SAD computation for rectangular region (fallback)

#[inline]
fn sad_rgb_rect_scalar(
    target: &[u8],
    current: &[u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
    best_so_far: Option<u64>,
) -> f64 {
    let mut sum: u64 = 0;
    const CHECK_INTERVAL: u32 = 8; // checks every 8 rows

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let idx = ((y * stride + x) * 4) as usize;

            let r_diff = (target[idx] as i32 - current[idx] as i32).abs() as u64;
            let g_diff = (target[idx + 1] as i32 - current[idx + 1] as i32).abs() as u64;
            let b_diff = (target[idx + 2] as i32 - current[idx + 2] as i32).abs() as u64;
            let a_diff = (target[idx + 3] as i32 - current[idx + 3] as i32).abs() as u64;

            sum += r_diff + g_diff + b_diff + a_diff;
        }

        // early-exit check every CHECK_INTERVAL rows
        if let Some(threshold) = best_so_far {
            if (y - y_min) % CHECK_INTERVAL == 0 && sum >= threshold {
                return u64::MAX as f64;
            }
        }
    }

    sum as f64
}

/// compute SAD over a rectangular region - dispatches to SIMD or scalar
/// rect is (x_min, y_min, x_max, y_max) in pixel coordinates (inclusive).
/// stride is the width of the full image in pixels.
#[inline]
pub fn sad_rgb_rect(
    target: &[u8],
    current: &[u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
    best_so_far: Option<u64>,
) -> f64 {
    profiling::scope!("sad_rgb_rect");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { sad_rgb_rect_avx2(target, current, x_min, y_min, x_max, y_max, stride, best_so_far) };
        }
    }

    sad_rgb_rect_scalar(target, current, x_min, y_min, x_max, y_max, stride, best_so_far)
}

/// copy a rectangular region from src to dst.
/// both buffers must have the same stride (width).
#[inline]
pub fn blit_rect(
    src: &[u8],
    dst: &mut [u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
) {
    profiling::scope!("blit_rect");

    // safety invariants: catch bbox union bugs (zero cost in release builds)
    debug_assert!(x_min <= x_max, "Invalid rect: x_min={} > x_max={}", x_min, x_max);
    debug_assert!(y_min <= y_max, "Invalid rect: y_min={} > y_max={}", y_min, y_max);

    for y in y_min..=y_max {
        let row_start = ((y * stride + x_min) * 4) as usize;
        let row_end = ((y * stride + x_max + 1) * 4) as usize;

        // safety: ensure we don't read/write past buffer end
        debug_assert!(row_end <= src.len(), "Buffer overrun: row_end={} > src.len()={}", row_end, src.len());
        debug_assert!(row_end <= dst.len(), "Buffer overrun: row_end={} > dst.len()={}", row_end, dst.len());

        dst[row_start..row_end].copy_from_slice(&src[row_start..row_end]);
    }
}

/// ---- tiled fitness ---------------------------------------------------------

/// tiled error cache for fast incremental fitness evaluation.
/// divides the image into NxN tiles and caches per-tile error sums.
/// when evaluating a mutation, only re-computes tiles overlapped by the bbox,
/// enabling early-exit as soon as accumulated error exceeds best_so_far.
#[derive(Clone)]
pub struct TileGrid {
    pub tile: u32,        // tile size in pixels (e.g., 32, 64, 128)
    pub tiles_x: u32,     // number of tiles horizontally
    pub tiles_y: u32,     // number of tiles vertically
    /// sum of abs diffs per tile at 1× (u64 to avoid overflow)
    pub errs: Vec<u64>,   // length = tiles_x * tiles_y
    /// cached total error (sum of all tiles) for O(1) full-image fitness queries
    pub total_err: u64,
}

impl TileGrid {
    /// create a new tile grid and compute initial per-tile errors.
    /// tile: tile size in pixels (recommend 32-128 depending on image size)
    /// w, h: image dimensions
    /// target, current: premultiplied RGBA buffers
    pub fn new(tile: u32, w: u32, h: u32, target: &[u8], current: &[u8]) -> Self {
        profiling::scope!("TileGrid::new");
        let tiles_x = (w + tile - 1) / tile;
        let tiles_y = (h + tile - 1) / tile;
        let errs = vec![0u64; (tiles_x * tiles_y) as usize];
        let mut tg = TileGrid { tile, tiles_x, tiles_y, errs, total_err: 0 };
        tg.recompute_all(w, h, target, current);
        tg
    }

    /// map pixel rect to tile indices (inclusive)
    #[inline]
    fn tile_rect(&self, _w: u32, _h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> (u32, u32, u32, u32) {
        let tx0 = (x0 / self.tile).min(self.tiles_x.saturating_sub(1));
        let ty0 = (y0 / self.tile).min(self.tiles_y.saturating_sub(1));
        let tx1 = (x1 / self.tile).min(self.tiles_x.saturating_sub(1));
        let ty1 = (y1 / self.tile).min(self.tiles_y.saturating_sub(1));
        (tx0, ty0, tx1, ty1)
    }

    /// recompute all tile errors from scratch (rare - only at init or full buffer changes)
    pub fn recompute_all(&mut self, w: u32, h: u32, target: &[u8], current: &[u8]) {
        profiling::scope!("TileGrid::recompute_all");
        let mut total = 0u64;
        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                let x0 = tx * self.tile;
                let y0 = ty * self.tile;
                let x1 = (x0 + self.tile - 1).min(w - 1);
                let y1 = (y0 + self.tile - 1).min(h - 1);
                let e = sad_rgb_rect(target, current, x0, y0, x1, y1, w, None) as u64;
                self.errs[(ty * self.tiles_x + tx) as usize] = e;
                total += e;
            }
        }
        self.total_err = total;
    }

    /// after accepting a mutation, update the cached tiles it touched.
    /// this keeps the cache in sync with the current buffer.
    /// maintains total_err incrementally for O(k) updates where k = affected tiles.
    pub fn accept_rect_update(&mut self, w: u32, h: u32, target: &[u8], current: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) {
        profiling::scope!("TileGrid::accept_rect_update");
        let (tx0, ty0, tx1, ty1) = self.tile_rect(w, h, x0, y0, x1, y1);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let idx = (ty * self.tiles_x + tx) as usize;
                let x_tile = tx * self.tile;
                let y_tile = ty * self.tile;
                let x_max = (x_tile + self.tile - 1).min(w - 1);
                let y_max = (y_tile + self.tile - 1).min(h - 1);

                // subtract old tile error from total
                let old_e = self.errs[idx];
                self.total_err -= old_e;

                // compute and cache new tile error
                let new_e = sad_rgb_rect(target, current, x_tile, y_tile, x_max, y_max, w, None) as u64;
                self.errs[idx] = new_e;

                // add new tile error to total
                self.total_err += new_e;
            }
        }
    }
}

//─────────────────────────────────────────────────────────────────────────────
// Resolution-Invariant Metrics (SAD/px, MSE, PSNR)
//─────────────────────────────────────────────────────────────────────────────

/// Number of channels in RGBA format (future-proof for potential format changes)
/// we're only using RGB for now so set it to 3
#[allow(dead_code)]
pub const FITNESS_CHANNELS: u32 = 3;
pub const FITNESS_CHANNELS_F64: f64 = 3.0;

/// Resolution-invariant SAD: normalizes absolute error by total pixel count.
/// Use this instead of raw SAD to compare images of different sizes.
#[inline]
pub fn sad_per_pixel(sad: f64, w: u32, h: u32) -> f64 {
    sad / ((w as f64) * (h as f64))
}

/// Pseudo-MSE derived from SAD (not true MSE).
/// NOTE: True MSE requires SSE (sum of squared errors). This is a placeholder
/// for API symmetry. Once SSE is available, replace with: SSE / (w * h * channels).
#[inline]
pub fn pseudo_mse_from_sad(sad: f64, w: u32, h: u32) -> f64 {
    sad_per_pixel(sad, w, h) / (FITNESS_CHANNELS_F64).max(1.0)
}

/// PSNR (Peak Signal-to-Noise Ratio) in decibels.
/// - `mse`: Mean Squared Error (or pseudo-MSE from SAD)
/// - `peak`: 255.0 for 8-bit images, 1.0 for normalized [0,1] range
/// Higher PSNR = better quality. Typical ranges:
///   - 30 dB = acceptable
///   - 35 dB = good
///   - 40+ dB = very good
#[inline]
pub fn psnr_from_mse(mse: f64, peak: f64) -> f64 {
    let mse = mse.max(1e-12); // avoid division by zero
    10.0 * ((peak * peak) / mse).log10()
}

/// Normalize a (possibly weighted) SAD so that downstream PSNR math remains
/// comparable whether luminance weighting is enabled or not.
#[inline]
pub fn normalized_sad_for_psnr(sad: f64, num_pixels: usize, avg_weight_q8: Option<u16>) -> f64 {
    let denom_unweighted = (num_pixels as f64) * 3.0;
    if let Some(avg_q8) = avg_weight_q8 {
        // Σw in 1.0 space = (avg_q8 * N) >> 8
        let sum_w_q8 = (avg_q8 as u64) * (num_pixels as u64);
        let sum_w_1_0 = (sum_w_q8 >> 8) as f64;
        let denom_weighted = sum_w_1_0 * FITNESS_CHANNELS_F64;
        if denom_weighted > 0.0 {
            // scale SAD so that old MSE≈SAD/(N*3) behaves like SADw/(Σw*3)
            sad * (denom_unweighted / denom_weighted)
        } else {
            sad
        }
    } else {
        sad
    }
}


/// Cached snapshot of resolution-invariant metrics.
/// Computed from raw SAD and image dimensions.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetricsSnapshot {
    pub sad_per_px: f64,   // SAD normalized by pixel count
    pub psnr: f64,         // PSNR in decibels
}


impl MetricsSnapshot {

    /// Build metrics directly from raw SAD + pixel count (unweighted path).
    #[inline]
    pub fn from_sad(sad: f64, num_pixels: usize, psnr_peak: f32) -> Self {
        let n = num_pixels as f64;
        let sad_per_px = sad / n;
        let channels = 3.0; // SAD is RGB-only, change to 4.0 if alpha is included in the future
        // Your current “pseudo-MSE from SAD” convention: treat L1/px/channel as if MSE.
        let pseudo_mse = (sad / (n * channels)).max(1e-12);
        let psnr = psnr_from_mse(pseudo_mse, psnr_peak as f64);
        Self { sad_per_px, psnr }
    }


    #[inline]
    pub fn from_sad_weighted_normalized(
        sad: f64,
        num_pixels: usize,
        avg_weight_q8: Option<u16>,
        psnr_peak: f32,
    ) -> Self {
        // sad_per_px from the ORIGINAL weighted SAD (preserve UI semantics)
        // This keeps the UI display consistent with the active fitness mode
        let sad_per_px = sad / (num_pixels as f64);

        // PSNR from the NORMALIZED SAD (for cross-run comparability)
        // Normalization ensures PSNR is comparable whether weighting is on or off
        let sad_for_psnr = normalized_sad_for_psnr(sad, num_pixels, avg_weight_q8);
        let channels = FITNESS_CHANNELS_F64; // 3.0 for RGB
        let pseudo_mse = (sad_for_psnr / ((num_pixels as f64) * channels)).max(1e-12);
        let psnr = psnr_from_mse(pseudo_mse, psnr_peak as f64);

        Self { sad_per_px, psnr }
    }
}