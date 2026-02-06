use std::env;
use eframe::egui;
use eframe::icon_data::IconDataExt;
use egui::IconData;
use crate::solver_lib::c_solver_app::SolverApp;
mod solver_lib;
mod shared_lib;

fn main() {
    let mut native_options = eframe::NativeOptions::default();

    let icon_png: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/icon.png"));
    let icon = eframe::icon_data::from_png_bytes(icon_png)
        .expect("assets/icon.png must be a valid PNG");


    native_options.viewport = egui::ViewportBuilder::default()
        .with_title("Neural Numbers - Solver")
        .with_inner_size([665.0, 480.0])
        .with_icon(icon);
    
    
    eframe::run_native(
        "Neural Numbers - Solver",
        native_options,
        Box::new(|cc| Ok(Box::new(SolverApp::new(cc)))),
    ).unwrap()
}