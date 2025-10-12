#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub hf_token: String
}

impl EnvConfig {
    pub fn init() -> EnvConfig {
        let hf_token = std::env::var("HF_TOKEN").expect("HF_TOKEN must be set");

        EnvConfig {
            hf_token
        }
    }
}