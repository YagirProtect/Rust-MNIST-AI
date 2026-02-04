use tch::nn;
use tch::nn::Module;
use crate::shared_lib::c_trainer_config::TrainerConfig;

pub fn build_model(vs: &nn::Path, cfg: &TrainerConfig) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(vs / "l1", cfg.image_dim, cfg.hidden, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "l2", cfg.hidden, cfg.labels, Default::default()))
}