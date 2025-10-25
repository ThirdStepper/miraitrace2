use eframe::egui;
use crate::app_types::FocusRegion;

/// Convert normalized error (0.0-1.0) to heatmap color (blue=low error/good, red=high error/bad)
fn error_to_heatmap_color(normalized_error: f32) -> egui::Color32 {
    let r = (normalized_error * 255.0).clamp(0.0, 255.0) as u8;
    let b = ((1.0 - normalized_error) * 255.0).clamp(0.0, 255.0) as u8;
    egui::Color32::from_rgba_unmultiplied(r, 0, b, 80)  // Semi-transparent overlay
}

/// Draw autofocus visualization overlay (tile grid + error heatmap)
pub fn draw_autofocus_overlay(
    response: &egui::Response,
    scale: f32,
    tex_size: egui::Vec2,
    painter: &egui::Painter,
    autofocus_tiles: &Option<Vec<(usize, f64, FocusRegion)>>,
    autofocus_active_indices: &Option<Vec<usize>>,
    show_tiles: bool,
    show_errors: bool,
) {
    profiling::scope!("draw_autofocus_overlay");
    // only draw if we have tile data
    let tiles = match autofocus_tiles {
        Some(t) => t,
        None => return,
    };

    // only draw if at least one visualization is enabled
    if !show_tiles && !show_errors {
        return;
    }

    // first tile has worst error (sorted)
    let rect = response.rect;
    let max_error = tiles[0].1.max(1.0);

    // build a fast lookup for active positions (positions in the sorted array)
    let mut is_active = vec![false; tiles.len()];
    if let Some(active) = autofocus_active_indices {
        for &p in active {
            if p < is_active.len() { is_active[p] = true; }
        }
    }

    for (pos, (/*tile_id*/_, sad_error, region)) in tiles.iter().enumerate() {
        // convert normalized region coords to screen coords
        let x1 = rect.min.x + region.left * tex_size.x * scale;
        let y1 = rect.min.y + region.top * tex_size.y * scale;
        let x2 = rect.min.x + region.right * tex_size.x * scale;
        let y2 = rect.min.y + region.bottom * tex_size.y * scale;

        let tile_rect = egui::Rect::from_min_max(
            egui::pos2(x1, y1),
            egui::pos2(x2, y2),
        );

        // show error heatmap: color tiles by normalized error
        if show_errors {
            let normalized = (sad_error / max_error) as f32;
            let color = error_to_heatmap_color(normalized);
            painter.rect_filled(tile_rect, 0.0, color);
        }

        // show tile grid: draw grid lines
        if show_tiles {
            // check if this tile is actually being used by the engine (using active_indices)
            // this correctly highlights multi-tile and probabilistic modes
            let is_focused = is_active.get(pos).copied().unwrap_or(false);

            let stroke = if is_focused {
                egui::Stroke::new(3.0, egui::Color32::RED)  // Highlight active tiles with thick red border
            } else {
                egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 100))
            };
            painter.rect_stroke(tile_rect, 0.0, stroke, egui::epaint::StrokeKind::Outside);
        }
    }
}

/// Draw red rectangle overlay showing the focus region
pub fn draw_region_overlay(
    response: &egui::Response,
    scale: f32,
    tex_size: egui::Vec2,
    region: FocusRegion,
    painter: &egui::Painter,
) {
    profiling::scope!("draw_region_overlay");
    let rect = response.rect;

    // convert normalized region coords to screen coords
    let x1 = rect.min.x + region.left * tex_size.x * scale;
    let y1 = rect.min.y + region.top * tex_size.y * scale;
    let x2 = rect.min.x + region.right * tex_size.x * scale;
    let y2 = rect.min.y + region.bottom * tex_size.y * scale;

    let overlay_rect = egui::Rect::from_min_max(
        egui::pos2(x1, y1),
        egui::pos2(x2, y2),
    );

    // draw red rectangle (matching Evolve's QPen color QColor(200,0,0,150))
    painter.rect_stroke(
        overlay_rect,
        0.0, // no corner rounding
        egui::Stroke::new(3.0, egui::Color32::from_rgba_unmultiplied(200, 0, 0, 150)),
        egui::epaint::StrokeKind::Outside,
    );
}
