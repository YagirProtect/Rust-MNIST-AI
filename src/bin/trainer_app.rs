use std::env;
use tch::{nn, Device};
use tch::nn::{Module, OptimizerConfig};
use crate::shared_lib::c_trainer_config::TrainerConfig;
use crate::shared_lib::f_ai_data::build_model;

mod trainer_lib;
mod shared_lib;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrainerConfig::new();

    let device = Device::cuda_if_available();
    println!("device: {:?}", device);


    let m = tch::vision::mnist::load_dir(env::current_dir().unwrap().join(&config.data_dir))?;

    let train_images = m.train_images.to_device(device);
    let train_labels = m.train_labels.to_device(device);
    let test_images = m.test_images.to_device(device);
    let test_labels = m.test_labels.to_device(device);

    let vs = nn::VarStore::new(device);
    let root = &vs.root();
    let model = build_model(root, &config);


    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..= config.epoch {
        let loss = model
            .forward(&train_images)
            .cross_entropy_for_logits(&train_labels);

        opt.backward_step(&loss);

        let acc = model
            .forward(&test_images)
            .accuracy_for_logits(&test_labels);

        let loss_value = loss.double_value(&[]);
        let acc_value  = acc.double_value(&[]);
        println!(
            "epoch {:3}/{:3} | loss {:8.5} | test acc {:5.2}%",
            epoch,
            config.epoch,
            loss_value,
            100.0 * acc_value
        );
    }
    vs.save(&config.out_path)?;
    println!("saved weights -> {}", config.out_path);

    return Ok(())
}


