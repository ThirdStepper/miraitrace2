mod app;
mod dna;
mod render;
mod mutate;
mod fitness;
mod engine;
mod analysis;
mod autofocus;
mod settings;

fn main() -> eframe::Result<()> {
    // Configure Rayon's *global* pool once at startup so worker threads get nice names
    // like "rayon-0", "rayon-1", … (harmless no-op if something already built it).
    let _ = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("rayon-{i}"))
        .build_global();

    // Tracy profiling is always-on when compiled with profile-with-tracy feature
    // Zero overhead when Tracy GUI is not connected

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "MiraiTrace2",
        native_options,
        Box::new(|cc| {
            Ok::<Box<dyn eframe::App>, Box<dyn std::error::Error + Send + Sync>>(
                Box::new(crate::app::MiraiApp::new(cc))
            )
        }),

    )

}