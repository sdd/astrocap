mod annotations;
mod app;
mod video;

use eframe::egui;
use signal_hook::{consts::*, iterator::Signals};
use std::panic;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

fn setup_signal_handlers() {
    let term = Arc::new(AtomicBool::new(false));

    // Handle various signals that could cause crashes
    let mut signals =
        Signals::new(&[SIGTERM, SIGINT, SIGQUIT]).expect("Failed to register signal handlers");

    let signals_term = term.clone();
    thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGTERM | SIGINT | SIGQUIT => {
                    eprintln!("Received termination signal: {}", sig);
                    signals_term.store(true, Ordering::Relaxed);
                    break;
                }
                _ => {
                    eprintln!("Received signal: {}", sig);
                }
            }
        }
    });
}

fn main() -> Result<(), eframe::Error> {
    // Set up signal handling first
    setup_signal_handlers();

    // Set up better panic handling
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("PANIC: {}", panic_info);

        if let Some(location) = panic_info.location() {
            eprintln!(
                "  Location: {}:{}:{}",
                location.file(),
                location.line(),
                location.column()
            );
        }

        if let Some(message) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!("  Message: {}", message);
        } else if let Some(message) = panic_info.payload().downcast_ref::<String>() {
            eprintln!("  Message: {}", message);
        }

        // Print backtrace if available
        eprintln!("Backtrace:");
        eprintln!("{}", std::backtrace::Backtrace::force_capture());
    }));

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(false)
        .compact()
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_maximized(true) // Start maximized
            .with_resizable(false) // Disable resizing to avoid cursor crashes
            .with_decorations(true), // Keep decorations for normal window controls
        hardware_acceleration: eframe::HardwareAcceleration::Off,
        centered: false, // Not needed when maximized

        ..Default::default()
    };

    eframe::run_native(
        "Astrocap Video Annotator",
        options,
        Box::new(|_cc| Ok(Box::new(app::AnnotatorApp::default()))),
    )
}
