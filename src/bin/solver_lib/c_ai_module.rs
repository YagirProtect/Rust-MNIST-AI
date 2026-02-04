use eframe::egui;
use tch::{nn, Device};
use crate::shared_lib::c_trainer_config::TrainerConfig;
use crate::shared_lib::f_ai_data::build_model;

pub struct AIModule{
    canvas_rect: Option<egui::Rect>,
    device: Device,
    vs: nn::VarStore,
    model: nn::Sequential,
    probs: [f32; 10],
}


impl AIModule{
    pub fn new(config: &TrainerConfig) -> Self {
        let device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(device);
        let root = &vs.root();
        let model = build_model(root, &config);
        vs.load(config.out_path.clone()).expect("Failed to load models/mnist.ot");


        Self{
            canvas_rect: None,
            device,
            vs,
            model,
            probs: [0.0; 10],
        }
    }
}