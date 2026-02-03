use std::env;
use std::io::BufReader;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TrainerConfig {
    epoch: usize,
    data_dir: String,
    out_path: String,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self{
            epoch: 20,
            data_dir: "data".to_string(),
            out_path: "models/mnist.ot".to_string()
        }
    }
}


impl TrainerConfig {
    pub fn new() -> Self {
        let file = env::current_dir().unwrap().join("config.json");

        if (!file.exists()) {
            let default = TrainerConfig::default();

            match serde_json::to_writer_pretty(&file, &default) {
                Ok(_) => {}
                Err(_) => {
                    println!("Error serializing config file");
                }
            }

            return default;
        }

        let mut config = TrainerConfig::default();
        let reader = BufReader::new(&file);

        *config = serde_json::from_reader(reader);
        return config
    }
}