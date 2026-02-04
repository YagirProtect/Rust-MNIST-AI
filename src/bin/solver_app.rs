use eframe::egui;
use crate::solver_lib::c_solver_app::SolverApp;
mod solver_lib;
mod shared_lib;

fn main() {
    let mut native_options = eframe::NativeOptions::default();

    native_options.viewport = egui::ViewportBuilder::default()
        .with_title("Neural Numbers - Solver")
        .with_inner_size([665.0, 480.0]);

    eframe::run_native(
        "Neural Numbers - Solver",
        native_options,
        Box::new(|cc| Ok(Box::new(SolverApp::new(cc)))),
    ).unwrap()
}