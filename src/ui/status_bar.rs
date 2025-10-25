use eframe::egui;
use crate::app_types::FocusRegion;
use crate::settings::AppSettings;

/// Render the bottom status bar panel
pub fn render_status_bar(
    ctx: &egui::Context,
    has_engine: bool,
    generation: u64,
    fitness: f32,
    triangles: usize,
    metrics: &crate::fitness::MetricsSnapshot,
    weighted_sad: Option<f64>,
    perceptual_k: Option<u16>,
    settings: &AppSettings,
    autofocus_tiles: &Option<Vec<(usize, f64, FocusRegion)>>,
    autofocus_active_indices: &Option<Vec<usize>>,
    target_dims: [usize; 2],
) {
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            // left: session info
            if has_engine {
                // Display metrics based on mode
                match settings.metrics_settings.mode {
                    crate::settings::MetricsMode::ResolutionInvariant => {
                        ui.label(format!("Gen: {} | PSNR*: {:.2} dB | SAD/px: {:.2} | Polys: {}",
                            generation, metrics.psnr, metrics.sad_per_px, triangles))
                            .on_hover_text("*PSNR approximated from L1 (SAD), not true L2");
                        ui.separator();
                        ui.weak(format!("({:.2}%)", fitness));
                    }
                    crate::settings::MetricsMode::Percentage => {
                        ui.label(format!("Gen: {} | Fitness: {:.2}% | Triangles: {}",
                            generation, fitness, triangles));
                        ui.separator();
                        ui.weak(format!("PSNR*: {:.2} dB, SAD/px: {:.2}",
                            metrics.psnr, metrics.sad_per_px))
                            .on_hover_text("*PSNR approximated from L1 (SAD), not true L2");
                    }
                }

                // Show weighted SAD in status bar when perceptual weighting is enabled
                if let (Some(wsad), Some(k)) = (weighted_sad, perceptual_k) {
                    ui.separator();
                    ui.weak(format!("Weighted: {:.0} (k={})", wsad, k))
                        .on_hover_text("Perceptual weighted SAD\nBright regions weighted more heavily");
                }
            } else {
                ui.label("No active session");
            }

            ui.separator();

            // center: Autofocus status
            if settings.autofocus_enabled && autofocus_tiles.is_some() {
                if let Some(tiles) = autofocus_tiles {
                    let worst_sad = tiles.first().map(|(_, err, _)| err).unwrap_or(&0.0);

                    // calculate proper percentage: normalize SAD against maximum possible error
                    // use actual tile area from FocusRegion (works for irregular quadtree/BSP tiles)
                    let default_region = FocusRegion::new(0.0, 1.0, 0.0, 1.0);
                    let worst_region = tiles.first().map(|(_, _, r)| r).unwrap_or(&default_region);
                    let tile_width_norm = worst_region.right - worst_region.left;
                    let tile_height_norm = worst_region.bottom - worst_region.top;
                    let tile_pixels = (tile_width_norm * target_dims[0] as f32 * tile_height_norm * target_dims[1] as f32) as u32;
                    // max SAD for RGBA (255 per channel × 4 channels) - matches sad_rgb_parallel computation
                    let max_error = tile_pixels as f64 * 255.0 * 4.0;
                    let error_percent = (worst_sad / max_error) * 100.0;

                    // display which tiles are actually active (based on mode)
                    let focus_description = if let Some(indices) = autofocus_active_indices {
                        if indices.len() == 1 {
                            let tile_id = tiles.get(indices[0]).map(|(id, _, _)| id).unwrap_or(&0);
                            format!("tile {} (error: {:.1}%)", tile_id, error_percent)
                        } else {
                            format!("{} tiles (worst error: {:.1}%)", indices.len(), error_percent)
                        }
                    } else {
                        format!("tile {} (error: {:.1}%)", tiles.first().map(|(idx, _, _)| idx).unwrap_or(&0), error_percent)
                    };

                    ui.label(format!("🎯 Autofocus: {} tiles, focusing {}",
                        tiles.len(), focus_description));
                }
            } else if settings.autofocus_enabled {
                ui.label("🎯 Autofocus: Initializing...");
            } else {
                ui.label("Autofocus: Off");
            }

            // right-aligned: image dimensions
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if target_dims[0] > 0 && target_dims[1] > 0 {
                    ui.label(format!("{}×{} px", target_dims[0], target_dims[1]));
                }
            });
        });
    });
}
