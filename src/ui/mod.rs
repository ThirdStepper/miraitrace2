// UI module organization
// Each submodule handles a specific aspect of the user interface

pub mod image_panel;
pub mod input;
pub mod overlays;
pub mod settings_window;
pub mod status_bar;
pub mod toolbar;

// Re-export commonly used functions for convenience
pub use image_panel::{aspect_fit, aspect_fit_interactive, render_central_panel};
pub use input::{handle_keyboard_input, handle_region_selection};
pub use overlays::{draw_autofocus_overlay, draw_region_overlay};
pub use settings_window::show_settings_window;
pub use status_bar::render_status_bar;
pub use toolbar::render_toolbar;
