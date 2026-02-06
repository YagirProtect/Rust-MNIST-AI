use std::{env, fs};
use std::path::Path;
use tch::{nn, Device, Kind, Tensor};
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
    finetune_on_mydata(&config);

    return Ok(())
}

fn finetune_on_mydata(config: &TrainerConfig){
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    let model = build_model(&vs.root(), &config);

    vs.load(&config.out_path).expect("load weights");

    let (x_cpu, y_cpu) = load_mydata("mydata", config);
    let x = x_cpu.to_device(device);
    let y = y_cpu.to_device(device);

    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    for epoch in 1..=350 {
        let logits = model.forward(&x);
        let loss = logits.cross_entropy_for_logits(&y);
        opt.backward_step(&loss);

        let acc = logits.accuracy_for_logits(&y);
        println!(
            "ft epoch {:02} | loss {:7.4} | acc {:5.2}%",
            epoch,
            loss.double_value(&[]),
            100.0 * acc.double_value(&[])
        );
    }

    vs.save(&config.out_path).expect("save finetuned");
}


pub fn load_mydata(dir: impl AsRef<Path>, config: &TrainerConfig) -> (Tensor, Tensor) {
    let dir = dir.as_ref();

    let mut images: Vec<f32> = Vec::new();
    let mut labels: Vec<i64> = Vec::new();

    for label in 0..10i64 {
        let sub = dir.join(label.to_string());
        let Ok(rd) = fs::read_dir(&sub) else { continue; };

        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("bin") {
                continue;
            }
            let Ok(bytes) = fs::read(&path) else { continue; };
            if bytes.len() != 28 * 28 {
                continue;
            }

            // u8 -> f32 (0..1)
            for &b in &bytes {
                images.push((b as f32) / 255.0);
            }
            labels.push(label);
        }
    }

    let n = labels.len() as i64;
    println!("Loaded mydata samples: {n}");

    // CNN: [N, 1, 28, 28]
    let x = Tensor::from_slice(&images)
        .to_kind(Kind::Float)
        .view([n, config.image_dim]);

    let y = Tensor::from_slice(&labels).to_kind(Kind::Int64);

    (x, y)
}

