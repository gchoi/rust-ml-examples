<<<<<<< HEAD
use std::fmt::Display;
use std::path::PathBuf;

use anyhow::bail;
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{Config as DebertaV2Config, DebertaV2NERModel};
use candle_transformers::models::debertav2::{DebertaV2SeqClassificationModel, Id2Label};
use candle_transformers::models::debertav2::{NERItem, TextClassificationItem};
use clap::{ArgGroup, Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{Encoding, PaddingParams, Tokenizer};

enum TaskType {
    Ner(Box<DebertaV2NERModel>),
    TextClassification(Box<DebertaV2SeqClassificationModel>),
}

#[derive(Parser, Debug, Clone, ValueEnum)]
enum ArgsTask {
    /// Named Entity Recognition
    Ner,

    /// Text Classification
    TextClassification,
}

impl Display for ArgsTask {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ArgsTask::Ner => write!(f, "ner"),
            ArgsTask::TextClassification => write!(f, "text-classification"),
        }
    }
=======
mod build;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::deepseek2::{DeepSeekV2, DeepSeekV2Config};

use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: DeepSeekV2,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: DeepSeekV2,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = {
            let temperature = temp.unwrap_or(0.);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<｜end▁of▁sentence｜>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <｜end▁of▁sentence｜> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "lite")]
    Lite,
    #[value(name = "lite-chat")]
    LiteChat,
    #[value(name = "coder-lite-chat")]
    CoderLiteChat,
    #[value(name = "v2")]
    V2,
    #[value(name = "v2-chat")]
    V2Chat,
>>>>>>> 5dea734 (add candle-devertav2)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
<<<<<<< HEAD
#[command(group(ArgGroup::new("model")
    .required(true)
    .args(&["model_id", "model_path"])))]
=======
>>>>>>> 5dea734 (add candle-devertav2)
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

<<<<<<< HEAD
    /// The model id to use from HuggingFace
    #[arg(long, requires_if("model_id", "revision"))]
    model_id: Option<String>,

    /// Revision of the model to use (default: "main")
    #[arg(long, default_value = "main")]
    revision: String,

    /// Specify a sentence to inference. Specify multiple times to inference multiple sentences.
    #[arg(long = "sentence", name="sentences", num_args = 1..)]
    sentences: Vec<String>,

    /// Use the pytorch weights rather than the by-default safetensors
    #[arg(long)]
    use_pth: bool,

    /// Perform a very basic benchmark on inferencing, using N number of iterations
    #[arg(long)]
    benchmark_iters: Option<usize>,

    /// Which task to run
    #[arg(long, default_value_t = ArgsTask::Ner)]
    task: ArgsTask,

    /// Use model from a specific directory instead of HuggingFace local cache.
    /// Using this ignores model_id and revision args.
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Pass in an Id2Label if the model config does not provide it, in JSON format. Example: --id2label='{"0": "True", "1": "False"}'
    #[arg(long)]
    id2label: Option<String>,
}

impl Args {
    fn build_model_and_tokenizer(
        &self,
    ) -> Result<(TaskType, DebertaV2Config, Tokenizer, Id2Label)> {
        let device = candle_examples::device(self.cpu)?;

        // Get files from either the HuggingFace API, or from a specified local directory.
        let (config_filename, tokenizer_filename, weights_filename) = {
            match &self.model_path {
                Some(base_path) => {
                    if !base_path.is_dir() {
                        bail!("Model path {} is not a directory.", base_path.display())
                    }

                    let config = base_path.join("config.json");
                    let tokenizer = base_path.join("tokenizer.json");
                    let weights = if self.use_pth {
                        base_path.join("pytorch_model.bin")
                    } else {
                        base_path.join("model.safetensors")
                    };
                    (config, tokenizer, weights)
                }
                None => {
                    let repo = Repo::with_revision(
                        self.model_id.as_ref().unwrap().clone(),
                        RepoType::Model,
                        self.revision.clone(),
                    );
                    let api = Api::new()?;
                    let api = api.repo(repo);
                    let config = api.get("config.json")?;
                    let tokenizer = api.get("tokenizer.json")?;
                    let weights = if self.use_pth {
                        api.get("pytorch_model.bin")?
                    } else {
                        api.get("model.safetensors")?
                    };
                    (config, tokenizer, weights)
                }
            }
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: DebertaV2Config = serde_json::from_str(&config)?;

        // Command-line id2label takes precedence. Otherwise, use model config's id2label.
        // If neither is specified, then we can't proceed.
        let id2label = if let Some(id2labelstr) = &self.id2label {
            serde_json::from_str(id2labelstr.as_str())?
        } else if let Some(id2label) = &config.id2label {
            id2label.clone()
        } else {
            bail!("Id2Label not found in the model configuration nor specified as a parameter")
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {e}")))?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        let vb = if self.use_pth {
            VarBuilder::from_pth(
                &weights_filename,
                candle_transformers::models::debertav2::DTYPE,
                &device,
            )?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_filename],
                    candle_transformers::models::debertav2::DTYPE,
                    &device,
                )?
            }
        };

        let vb = vb.set_prefix("deberta");

        match self.task {
            ArgsTask::Ner => Ok((
                TaskType::Ner(DebertaV2NERModel::load(vb, &config, Some(id2label.clone()))?.into()),
                config,
                tokenizer,
                id2label,
            )),
            ArgsTask::TextClassification => Ok((
                TaskType::TextClassification(
                    DebertaV2SeqClassificationModel::load(vb, &config, Some(id2label.clone()))?
                        .into(),
                ),
                config,
                tokenizer,
                id2label,
            )),
        }
    }
}

fn get_device(model_type: &TaskType) -> &Device {
    match model_type {
        TaskType::Ner(ner_model) => &ner_model.device,
        TaskType::TextClassification(classification_model) => &classification_model.device,
    }
}

struct ModelInput {
    encoding: Vec<Encoding>,
    input_ids: Tensor,
    attention_mask: Tensor,
    token_type_ids: Tensor,
=======
    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    /// The model size to use.
    #[arg(long, default_value = "lite")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
>>>>>>> 5dea734 (add candle-devertav2)
}

fn main() -> Result<()> {
    let args = Args::parse();

<<<<<<< HEAD
    let model_load_time = std::time::Instant::now();
    let (task_type, _model_config, tokenizer, id2label) = args.build_model_and_tokenizer()?;

    println!(
        "Loaded model and tokenizers in {:?}",
        model_load_time.elapsed()
    );

    let device = get_device(&task_type);

    let tokenize_time = std::time::Instant::now();

    let model_input: ModelInput = {
        let tokenizer_encodings = tokenizer
            .encode_batch(args.sentences, true)
            .map_err(E::msg)?;

        let mut encoding_stack: Vec<Tensor> = Vec::default();
        let mut attention_mask_stack: Vec<Tensor> = Vec::default();
        let mut token_type_id_stack: Vec<Tensor> = Vec::default();

        for encoding in &tokenizer_encodings {
            encoding_stack.push(Tensor::new(encoding.get_ids(), device)?);
            attention_mask_stack.push(Tensor::new(encoding.get_attention_mask(), device)?);
            token_type_id_stack.push(Tensor::new(encoding.get_type_ids(), device)?);
        }

        ModelInput {
            encoding: tokenizer_encodings,
            input_ids: Tensor::stack(&encoding_stack[..], 0)?,
            attention_mask: Tensor::stack(&attention_mask_stack[..], 0)?,
            token_type_ids: Tensor::stack(&token_type_id_stack[..], 0)?,
        }
    };

    println!(
        "Tokenized and loaded inputs in {:?}",
        tokenize_time.elapsed()
    );

    match task_type {
        TaskType::Ner(ner_model) => {
            if let Some(num_iters) = args.benchmark_iters {
                create_benchmark(num_iters, model_input)(
                    |input_ids, token_type_ids, attention_mask| {
                        ner_model.forward(input_ids, Some(token_type_ids), Some(attention_mask))?;
                        Ok(())
                    },
                )?;

                std::process::exit(0);
            }

            let inference_time = std::time::Instant::now();
            let logits = ner_model.forward(
                &model_input.input_ids,
                Some(model_input.token_type_ids),
                Some(model_input.attention_mask),
            )?;

            println!("Inferenced inputs in {:?}", inference_time.elapsed());

            let max_scores_vec = softmax(&logits, 2)?.max(2)?.to_vec2::<f32>()?;
            let max_indices_vec: Vec<Vec<u32>> = logits.argmax(2)?.to_vec2()?;
            let input_ids = model_input.input_ids.to_vec2::<u32>()?;
            let mut results: Vec<Vec<NERItem>> = Default::default();

            for (input_row_idx, input_id_row) in input_ids.iter().enumerate() {
                let mut current_row_result: Vec<NERItem> = Default::default();
                let current_row_encoding = model_input.encoding.get(input_row_idx).unwrap();
                let current_row_tokens = current_row_encoding.get_tokens();
                let current_row_max_scores = max_scores_vec.get(input_row_idx).unwrap();

                for (input_id_idx, _input_id) in input_id_row.iter().enumerate() {
                    // Do not include special characters in output
                    if current_row_encoding.get_special_tokens_mask()[input_id_idx] == 1 {
                        continue;
                    }

                    let max_label_idx = max_indices_vec
                        .get(input_row_idx)
                        .unwrap()
                        .get(input_id_idx)
                        .unwrap();

                    let label = id2label.get(max_label_idx).unwrap().clone();

                    // Do not include those labeled as "O" ("Other")
                    if label == "O" {
                        continue;
                    }

                    current_row_result.push(NERItem {
                        entity: label,
                        word: current_row_tokens[input_id_idx].clone(),
                        score: current_row_max_scores[input_id_idx],
                        start: current_row_encoding.get_offsets()[input_id_idx].0,
                        end: current_row_encoding.get_offsets()[input_id_idx].1,
                        index: input_id_idx,
                    });
                }

                results.push(current_row_result);
            }

            println!("\n{results:?}");
        }

        TaskType::TextClassification(classification_model) => {
            let inference_time = std::time::Instant::now();
            let logits = classification_model.forward(
                &model_input.input_ids,
                Some(model_input.token_type_ids),
                Some(model_input.attention_mask),
            )?;

            println!("Inferenced inputs in {:?}", inference_time.elapsed());

            let predictions = logits.argmax(1)?.to_vec1::<u32>()?;
            let scores = softmax(&logits, 1)?.max(1)?.to_vec1::<f32>()?;
            let mut results = Vec::<TextClassificationItem>::default();

            for (idx, prediction) in predictions.iter().enumerate() {
                results.push(TextClassificationItem {
                    label: id2label[prediction].clone(),
                    score: scores[idx],
                });
            }

            println!("\n{results:?}");
        }
    }
    Ok(())
}

fn create_benchmark<F>(
    num_iters: usize,
    model_input: ModelInput,
) -> impl Fn(F) -> Result<(), candle_core::Error>
where
    F: Fn(&Tensor, Tensor, Tensor) -> Result<(), candle_core::Error>,
{
    move |code: F| -> Result<(), candle_core::Error> {
        println!("Running {num_iters} iterations...");
        let mut durations = Vec::with_capacity(num_iters);
        for _ in 0..num_iters {
            let token_type_ids = model_input.token_type_ids.clone();
            let attention_mask = model_input.attention_mask.clone();
            let start = std::time::Instant::now();
            code(&model_input.input_ids, token_type_ids, attention_mask)?;
            let duration = start.elapsed();
            durations.push(duration.as_nanos());
        }

        let min_time = *durations.iter().min().unwrap();
        let max_time = *durations.iter().max().unwrap();
        let avg_time = durations.iter().sum::<u128>() as f64 / num_iters as f64;

        println!("Min time: {:.3} ms", min_time as f64 / 1_000_000.0);
        println!("Avg time: {:.3} ms", avg_time / 1_000_000.0);
        println!("Max time: {:.3} ms", max_time as f64 / 1_000_000.0);
        Ok(())
    }
}
=======
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => match args.which {
            Which::CoderLiteChat => "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct".to_string(),
            Which::LiteChat => "deepseek-ai/DeepSeek-V2-Lite-Chat".to_string(),
            Which::Lite => "deepseek-ai/DeepSeek-V2-Lite".to_string(),
            Which::V2 => "deepseek-ai/DeepSeek-V2".to_string(),
            Which::V2Chat => "deepseek-ai/DeepSeek-V2-Chat".to_string(),
        },
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let filenames = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config: DeepSeekV2Config = {
        let config_file = repo.get("config.json")?;
        serde_json::from_slice(&std::fs::read(config_file)?)?
    };
    let device = candle_examples::device(args.cpu)?;
    let (model, device) = {
        let dtype = if device.is_cpu() {
            DType::F16
        } else {
            DType::BF16
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = DeepSeekV2::new(&config, vb)?;
        (model, device)
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
>>>>>>> 5dea734 (add candle-devertav2)
