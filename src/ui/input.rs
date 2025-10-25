use eframe::egui;
use std::sync::mpsc;

use crate::app_types::{EngineCommand, FocusRegion};
use crate::settings::AppSettings;

/// Handle keyboard shortcuts
pub fn handle_keyboard_input(
    ctx: &egui::Context,
    has_engine: bool,
    running: &mut bool,
    show_settings: &mut bool,
    settings: &mut AppSettings,
    command_tx: &Option<mpsc::Sender<EngineCommand>>,
    on_load_image: &mut bool,
    on_save_settings: &mut bool,
) {
    ctx.input(|i| {
        // space: toggle run/pause
        if i.key_pressed(egui::Key::Space) && has_engine {
            *running = !*running;
            if let Some(tx) = command_tx {
                let cmd = if *running {
                    EngineCommand::Start
                } else {
                    EngineCommand::Pause
                };
                let _ = tx.send(cmd);
            }
        }

        // f: toggle autofocus
        if i.key_pressed(egui::Key::F) && has_engine {
            settings.autofocus_enabled = !settings.autofocus_enabled;
            if let Some(tx) = command_tx {
                let pack = crate::settings::AutofocusPack::from(&*settings);
                let _ = tx.send(EngineCommand::UpdateAutofocusSettings(pack));
            }
        }

        // g: toggle grid visibility (only when autofocus is enabled)
        if i.key_pressed(egui::Key::G) && has_engine && settings.autofocus_enabled {
            settings.autofocus_show_tiles = !settings.autofocus_show_tiles;
        }

        // h: toggle heatmap (only when autofocus is enabled)
        if i.key_pressed(egui::Key::H) && has_engine && settings.autofocus_enabled {
            settings.autofocus_show_errors = !settings.autofocus_show_errors;
        }

        // s: toggle settings window
        if i.key_pressed(egui::Key::S) && !i.modifiers.ctrl {
            *show_settings = !*show_settings;
        }

        // ctrl+O: open image
        if i.key_pressed(egui::Key::O) && i.modifiers.ctrl {
            *on_load_image = true;
        }

        // ctrl+s: save settings
        if i.key_pressed(egui::Key::S) && i.modifiers.ctrl {
            *on_save_settings = true;
        }

        // ctrl+e: Export SVG
        if i.key_pressed(egui::Key::E) && i.modifiers.ctrl {
            // TODO: export genome as SVG
        }
    });
}

/// Handle mouse interaction for region selection
pub fn handle_region_selection(
    response: &egui::Response,
    scale: f32,
    tex_size: egui::Vec2,
    drag_start: &mut Option<egui::Pos2>,
    focus_region: &mut Option<FocusRegion>,
    command_tx: &Option<mpsc::Sender<EngineCommand>>,
) {
    profiling::scope!("handle_region_selection");
    // left mouse button - start drag
    if response.drag_started_by(egui::PointerButton::Primary) {
        if let Some(pos) = response.interact_pointer_pos() {
            let rect = response.rect;
            // convert to image space (relative to top-left of image)
            let img_pos = egui::pos2(pos.x - rect.min.x, pos.y - rect.min.y);
            *drag_start = Some(img_pos);
        }
    }

    // left mouse button - dragging
    if response.dragged_by(egui::PointerButton::Primary) {
        if let (Some(start), Some(current)) = (*drag_start, response.interact_pointer_pos()) {
            let rect = response.rect;
            let img_current = egui::pos2(current.x - rect.min.x, current.y - rect.min.y);

            // convert to normalized coordinates (0.0-1.0)
            let start_norm = egui::pos2(
                (start.x / (tex_size.x * scale)).clamp(0.0, 1.0),
                (start.y / (tex_size.y * scale)).clamp(0.0, 1.0),
            );
            let current_norm = egui::pos2(
                (img_current.x / (tex_size.x * scale)).clamp(0.0, 1.0),
                (img_current.y / (tex_size.y * scale)).clamp(0.0, 1.0),
            );

            // create focus region from drag
            let left = start_norm.x.min(current_norm.x);
            let right = start_norm.x.max(current_norm.x);
            let top = start_norm.y.min(current_norm.y);
            let bottom = start_norm.y.max(current_norm.y);

            *focus_region = Some(FocusRegion::new(left, right, top, bottom));

            // send command to engine
            if let Some(tx) = command_tx {
                let _ = tx.send(EngineCommand::SetFocusRegion(*focus_region));
            }
        }
    }

    // left mouse button - end drag
    if response.drag_stopped() {
        *drag_start = None;
    }

    // right mouse button - reset region
    if response.clicked_by(egui::PointerButton::Secondary) {
        *focus_region = None;
        *drag_start = None;

        // send command to engine to clear region
        if let Some(tx) = command_tx {
            let _ = tx.send(EngineCommand::SetFocusRegion(None));
        }
    }
}
