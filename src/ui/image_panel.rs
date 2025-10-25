use eframe::egui::{self, Image, TextureHandle};
use std::sync::mpsc;

use crate::app_types::{EngineCommand, FocusRegion};
use crate::settings::AppSettings;

/// Draw a texture scaled to fit within the available space preserving aspect ratio
pub fn aspect_fit(ui: &mut egui::Ui, tex: &TextureHandle) {
    let avail = ui.available_size();
    let tex_size = tex.size_vec2();
    let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).max(0.0);
    let draw_size = tex_size * scale;
    ui.add(Image::new(tex).fit_to_exact_size(draw_size));
}

/// Draw texture with mouse interaction for region selection (returns response and scale)
pub fn aspect_fit_interactive(ui: &mut egui::Ui, tex: &TextureHandle) -> (egui::Response, f32, egui::Vec2) {
    let avail = ui.available_size();
    let tex_size = tex.size_vec2();
    let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).max(0.0);
    let draw_size = tex_size * scale;
    let response = ui.add(Image::new(tex).fit_to_exact_size(draw_size).sense(egui::Sense::click_and_drag()));
    (response, scale, tex_size)
}

/// Render the central panel (welcome screen or side-by-side image view)
pub fn render_central_panel(
    ctx: &egui::Context,
    target_tex: &Option<TextureHandle>,
    current_tex: &Option<TextureHandle>,
    // State for region selection
    drag_start: &mut Option<egui::Pos2>,
    focus_region: &mut Option<FocusRegion>,
    command_tx: &Option<mpsc::Sender<EngineCommand>>,
    // Autofocus visualization
    autofocus_tiles: &Option<Vec<(usize, f64, FocusRegion)>>,
    autofocus_active_indices: &Option<Vec<usize>>,
    settings: &AppSettings,
    on_load_image: &mut bool,
) {
    egui::CentralPanel::default().show(ctx, |ui| {
        if target_tex.is_none() && current_tex.is_none() {
            // show enhanced welcome screen
            render_welcome_screen(ui, ctx, on_load_image);
        } else {
            // show side-by-side comparison
            render_image_comparison(
                ui,
                target_tex,
                current_tex,
                drag_start,
                focus_region,
                command_tx,
                autofocus_tiles,
                autofocus_active_indices,
                settings,
            );
        }
    });
}

/// Render the welcome screen when no image is loaded
fn render_welcome_screen(ui: &mut egui::Ui, _ctx: &egui::Context, on_load_image: &mut bool) {
    ui.vertical_centered(|ui| {
        ui.add_space(80.0);

        // Title and tagline
        ui.heading(egui::RichText::new("MiraiTrace2")
            .size(32.0)
            .color(egui::Color32::from_rgb(100, 150, 255)));
        ui.label(egui::RichText::new("Genetic Image Vectorization")
            .size(16.0)
            .color(egui::Color32::from_rgb(150, 150, 150)));
        ui.add_space(30.0);

        // drag-drop zone visual
        let available = ui.available_rect_before_wrap();
        let center = available.center();
        let drop_zone = ui.allocate_rect(
            egui::Rect::from_center_size(
                center,
                egui::vec2(400.0, 150.0)
            ),
            egui::Sense::hover()
        );

        ui.painter().rect_stroke(
            drop_zone.rect,
            8.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 255).linear_multiply(0.5)),
            egui::epaint::StrokeKind::Outside,
        );

        let mut child_ui = ui.new_child(egui::UiBuilder::new().max_rect(drop_zone.rect));
        child_ui.vertical_centered(|ui| {
            ui.add_space(35.0);
            ui.label(egui::RichText::new("📁")
                .size(48.0));
            ui.add_space(10.0);
            ui.label(egui::RichText::new("Drag & Drop an image here")
                .size(18.0));
            ui.label("or");
            if ui.button(egui::RichText::new("Open Image… (Ctrl+O)")
                .size(14.0)).clicked() {
                *on_load_image = true;
            }
        });

        ui.add_space(40.0);

        // quick tips section
        ui.group(|ui| {
            ui.set_max_width(500.0);
            ui.label(egui::RichText::new("💡 Quick Tips")
                .size(16.0)
                .strong());
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.label("•");
                ui.label("Use autofocus (F) for 2-4× speedup on complex images");
            });
            ui.horizontal(|ui| {
                ui.label("•");
                ui.label("Left-drag on the evolved image to manually focus on regions");
            });
            ui.horizontal(|ui| {
                ui.label("•");
                ui.label("Toggle grid (G) and heatmap (H) to visualize tile errors");
            });
            ui.horizontal(|ui| {
                ui.label("•");
                ui.label("Press Space to pause/resume evolution at any time");
            });
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Keyboard Shortcuts:")
                    .strong());
            });
            ui.add_space(5.0);
            ui.horizontal(|ui| {
                ui.monospace("Space");
                ui.label("→ Play/Pause");
                ui.separator();
                ui.monospace("F");
                ui.label("→ Toggle Autofocus");
                ui.separator();
                ui.monospace("S");
                ui.label("→ Settings");
            });
            ui.horizontal(|ui| {
                ui.monospace("G");
                ui.label("→ Toggle Grid");
                ui.separator();
                ui.monospace("H");
                ui.label("→ Toggle Heatmap");
                ui.separator();
                ui.monospace("Ctrl+S");
                ui.label("→ Save Settings");
            });
        });
    });
}

/// Render the side-by-side image comparison view
fn render_image_comparison(
    ui: &mut egui::Ui,
    target_tex: &Option<TextureHandle>,
    current_tex: &Option<TextureHandle>,
    drag_start: &mut Option<egui::Pos2>,
    focus_region: &mut Option<FocusRegion>,
    command_tx: &Option<mpsc::Sender<EngineCommand>>,
    autofocus_tiles: &Option<Vec<(usize, f64, FocusRegion)>>,
    autofocus_active_indices: &Option<Vec<usize>>,
    settings: &AppSettings,
) {
    ui.columns(2, |cols| {
        // left: current vectorized render (with mouse interaction for region selection)
        cols[0].vertical_centered(|ui| {
            ui.heading("Current (Evolved)");
        });
        if let Some(current) = current_tex {
            let (response, scale, tex_size) = aspect_fit_interactive(&mut cols[0], current);

            // handle mouse interaction for region selection
            crate::ui::input::handle_region_selection(
                &response,
                scale,
                tex_size,
                drag_start,
                focus_region,
                command_tx,
            );

            // draw autofocus tile overlay (grid + heatmap)
            let painter = cols[0].painter();
            crate::ui::overlays::draw_autofocus_overlay(
                &response,
                scale,
                tex_size,
                painter,
                autofocus_tiles,
                autofocus_active_indices,
                settings.autofocus_show_tiles,
                settings.autofocus_show_errors,
            );

            // draw manual focus region overlay if active (drawn on top of autofocus)
            if let Some(region) = focus_region {
                crate::ui::overlays::draw_region_overlay(&response, scale, tex_size, *region, painter);
            }
        } else {
            cols[0].centered_and_justified(|ui| {
                ui.label("Rendering...");
            });
        }

        // right: target image
        cols[1].vertical_centered(|ui| {
            ui.heading("Target (Original)");
        });
        if let Some(target) = target_tex {
            aspect_fit(&mut cols[1], target);
        }
    });
}
