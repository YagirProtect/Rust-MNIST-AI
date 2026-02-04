use tch::nn;
use tch::nn::Module;
use crate::shared_lib::c_trainer_config::TrainerConfig;

pub fn build_model(vs: &nn::Path, cfg: &TrainerConfig) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(vs / "l1", cfg.image_dim, cfg.hidden, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "l2", cfg.hidden, cfg.labels, Default::default()))
}

pub fn build_model_cnn(vs: &nn::Path) -> nn::Sequential {
    let c1 = nn::ConvConfig { padding: 1, ..Default::default() };
    let c2 = nn::ConvConfig { padding: 1, ..Default::default() };

    nn::seq()
        .add(nn::conv2d(vs / "c1", 1, 16, 3, c1))
        .add_fn(|x| x.relu())
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs / "c2", 16, 32, 3, c2))
        .add_fn(|x| x.relu())
        .add_fn(|x| x.max_pool2d_default(2))
        .add_fn(|x| x.view([-1, 32 * 7 * 7]))
        .add(nn::linear(vs / "fc1", 32 * 7 * 7, 128, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "fc2", 128, 10, Default::default()))
}