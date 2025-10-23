mod app;
mod dna;
mod render;
mod mutate;
mod fitness;
mod engine;
mod analysis;
mod autofocus;
mod settings;
mod geom;

fn main() -> eframe::Result<()> {
    // configure Rayon's global thread pool once at startup so worker threads get nice names like "rayon-0".
    let _ = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("rayon-{i}"))
        .build_global();

    // tracy profiling is always-on when compiled with profile-with-tracy feature
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