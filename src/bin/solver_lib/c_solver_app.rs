use crate::shared_lib::c_trainer_config::TrainerConfig;
use crate::solver_lib::c_ai_module::AIModule;
use crate::solver_lib::c_painter_module::PainterModule;
use eframe::egui;

pub struct SolverApp {
    config: TrainerConfig,
    painter_data: PainterModule,
    ai_module: AIModule


}

impl SolverApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let config = TrainerConfig::new();

        Self{
            painter_data: PainterModule::default(),
            ai_module: AIModule::new(&config),
            config,
        }
    }
}


impl eframe::App for SolverApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Draw a digit");

            self.painter_data.draw_painter_panel(ui);

            ui.add_space(10.0);
            ui.label("Tip: draw big and centered.");
        });




        egui::SidePanel::right("right")
            .min_width(160.0).show(ctx, |ui| {
            ui.add(egui::Slider::new(&mut self.painter_data.brush_size, 3.0..=10.0).text("Brush size"));
            if (ui.button("Clear").clicked()){
                self.painter_data.clear();
            }

            ui.add_space(100.0);
            ui.separator();
        });
    }
}

