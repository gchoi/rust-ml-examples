# candle-chinese-clip

Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
pairs of images with related texts. This one is trained using in chinese instead of english.

## Running on CPU

```bash
# CPU
$ cargo run --release -- \
  --images "../assets/stable-diffusion-xl.jpg","../assets/bike.jpg" \
  --cpu \
  --sequences "一场自行车比赛","两只猫的照片","一个机器人拿着蜡烛"

> 2025-10-14T00:08:07.121804Z  INFO chinese_clip: Transformer loaded. 
> 2025-10-14T00:08:07.132690Z  INFO chinese_clip: Images loaded. 
> 2025-10-14T00:08:07.141705Z  INFO chinese_clip: Computing ... 
> 2025-10-14T00:08:07.574165Z  INFO chinese_clip: 
> 
> Results for image: ../assets/stable-diffusion-xl.jpg
> 
> 2025-10-14T00:08:07.574184Z  INFO chinese_clip: Probability: 0.0000% Text: 一场自行车比赛 
> 2025-10-14T00:08:07.574187Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-10-14T00:08:07.574189Z  INFO chinese_clip: Probability: 100.0000% Text: 一个机器人拿着蜡烛 
> 2025-10-14T00:08:07.574192Z  INFO chinese_clip: 
> 
> Results for image: ../assets/bike.jpg
> 
> 2025-10-14T00:08:07.574194Z  INFO chinese_clip: Probability: 100.0000% Text: 一场自行车比赛 
> 2025-10-14T00:08:07.574196Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-10-14T00:08:07.574197Z  INFO chinese_clip: Probability: 0.0000% Text: 一个机器人拿着蜡烛  
```

## Running on CUDA

```bash 
$ cargo run --features cuda --release -- \
  --images "../assets/stable-diffusion-xl.jpg","../assets/bike.jpg" \
  --cpu \
  --sequences "一场自行车比赛","两只猫的照片","一个机器人拿着蜡烛"

> 2025-10-14T00:17:45.002587Z  INFO chinese_clip: Transformer loaded. 
> 2025-10-14T00:17:45.044261Z  INFO chinese_clip: Images loaded. 
> 2025-10-14T00:17:48.029069Z  INFO chinese_clip: 
> 
> Results for image: ../assets/stable-diffusion-xl.jpg
> 
> 2025-10-14T00:17:48.029112Z  INFO chinese_clip: Probability: 0.0000% Text: 一场自行车比赛 
> 2025-10-14T00:17:48.029126Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-10-14T00:17:48.029150Z  INFO chinese_clip: Probability: 100.0000% Text: 一个机器人拿着蜡烛 
> 2025-10-14T00:17:48.029177Z  INFO chinese_clip: 
> 
> Results for image: ../assets/bike.jpg
> 
> 2025-10-14T00:17:48.029185Z  INFO chinese_clip: Probability: 100.0000% Text: 一场自行车比赛 
> 2025-10-14T00:17:48.029194Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-10-14T00:17:48.029199Z  INFO chinese_clip: Probability: 0.0000% Text: 一个机器人拿着蜡烛  
```


## Running on Metal

```bash 
$ cargo run --features metal --release -- \
  --images "../assets/stable-diffusion-xl.jpg","../assets/bike.jpg" \
  --cpu \
  --sequences "一场自行车比赛","两只猫的照片","一个机器人拿着蜡烛"

> 2025-10-14T00:11:26.553018Z  INFO chinese_clip: Transformer loaded. 
> 2025-10-14T00:11:26.563501Z  INFO chinese_clip: Images loaded. 
> 2025-10-14T00:11:26.572758Z  INFO chinese_clip: Computing ... 
> 2025-10-14T00:11:26.987410Z  INFO chinese_clip: 
> 
> Results for image: ../assets/stable-diffusion-xl.jpg
> 
> 2025-10-14T00:11:26.987433Z  INFO chinese_clip: Probability: 0.0000% Text: 一场自行车比赛 
> 2025-10-14T00:11:26.987436Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-10-14T00:11:26.987438Z  INFO chinese_clip: Probability: 100.0000% Text: 一个机器人拿着蜡烛 
> 2025-10-14T00:11:26.987441Z  INFO chinese_clip: 
> 
> Results for image: ../assets/bike.jpg
> 
> 2025-10-14T00:11:26.987443Z  INFO chinese_clip: Probability: 100.0000% Text: 一场自行车比赛 
> 2025-10-14T00:11:26.987445Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-10-14T00:11:26.987446Z  INFO chinese_clip: Probability: 0.0000% Text: 一个机器人拿着蜡烛 
```