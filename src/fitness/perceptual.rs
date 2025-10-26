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
