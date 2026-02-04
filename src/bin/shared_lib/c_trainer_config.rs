use std::env;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TrainerConfig {
    pub epoch: usize,
    pub data_dir: String,
    pub out_path: String,

    pub image_dim: i64,
    pub hidden: i64,
    pub labels: i64

}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self{
            epoch: 20,
            data_dir: "data/".to_string(),
            out_path: "models/mnist.ot".to_string(),

            image_dim: 784,
            hidden: 128,
            labels: 10,
        }
    }
}


impl TrainerConfig {
    pub fn new() -> Self {
        let file = env::current_dir().unwrap().join("config.json");

        if (!file.exists()) {
            let default = TrainerConfig::default();
            Self::save(&file, default);
        }

        let file_open = std::fs::File::open(&file).expect("Failed to create json file");
        let reader = BufReader::new(&file_open);

        let mut cfg = serde_json::from_reader(reader).unwrap_or_default();

        cfg = Self::save(&file, cfg);

        return cfg
    }

    fn save(file: &PathBuf, data: TrainerConfig) -> TrainerConfig {
        
        let file = std::fs::File::create(&file).expect("Failed to create json file");
        let mut w = BufWriter::new(file);


        match serde_json::to_writer_pretty(&mut w, &data) {
            Ok(_) => {}
            Err(_) => {
                println!("Error serializing config file");
            }
        }
        return data;
    }
}