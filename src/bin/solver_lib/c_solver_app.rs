use std::env;
use crate::shared_lib::c_trainer_config::TrainerConfig;
use crate::solver_lib::c_ai_module::AIModule;
use crate::solver_lib::c_painter_module::PainterModule;
use eframe::egui;
use crate::solver_lib::f_utils::{predict_from_canvas, save_bmp_gray_f32, save_sample_u8};

pub struct SolverApp {
    config: TrainerConfig,
    painter_data: PainterModule,
    ai_module: AIModule,

    selected_label: u8,
}

impl SolverApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let config = TrainerConfig::new();

        Self{
            painter_data: PainterModule::default(),
            ai_module: AIModule::new(&config),
            config,
            selected_label: 0,
        }
    }
}


impl eframe::App for SolverApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Draw a digit");

            if (self.painter_data.draw_painter_panel(ui)){
                predict_from_canvas(&self.painter_data, &mut self.ai_module, &self.config);
            }

            ui.add_space(10.0);
            ui.label("Tip: draw big and centered.");
        });




        egui::SidePanel::right("right")
            .min_width(160.0).show(ctx, |ui| {

            ui.add(egui::Slider::new(&mut self.painter_data.brush_size, 3.0..=20.0).text("Brush size"));
            if (ui.button("Clear").clicked()){
                self.painter_data.clear();
            }

            // if ui.button("Save 28x28 BMP").clicked() {
            //     if let Some(pix) = &self.ai_module.last_28_pixels {
            //         let filename = env::current_dir().unwrap().join("debug_28x28.bmp");
            //         match save_bmp_gray_f32(filename, 28, 28, pix, false) {
            //             Ok(_) => {},
            //             Err(e) => {},
            //         }
            //     } else {
            //
            //     }
            // }

            egui::ComboBox::from_label("Label")
                .selected_text(self.selected_label.to_string())
                .show_ui(ui, |ui| {
                    for d in 0u8..=9 {
                        ui.selectable_value(&mut self.selected_label, d, d.to_string());
                    }
                });

            if (ui.button("Save correct answer").clicked()){

                if let Some(v) = self.ai_module.last_28_pixels.as_ref() {
                    if let Err(e) = save_sample_u8(v, self.selected_label) {
                        eprintln!("Can't save correct answer: {e}");
                    }
                }


            }


            ui.add_space(100.0);
            ui.separator();

            ui.heading("AI Thinking: ");
            for (i, &val) in self.ai_module.probs.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);

                let size = 12.0 + t * 16.0;

                let r = (140.0 * (1.0 - t)) as u8;
                let g = (80.0 + 175.0 * t) as u8;
                let b = (140.0 * (1.0 - t)) as u8;

                let color = egui::Color32::from_rgb(r, g, b);

                let text = format!("{i}: {:5.1}%", t * 100.0);
                ui.label(egui::RichText::new(text).size(size).color(color));
            }
        });
    }
}

