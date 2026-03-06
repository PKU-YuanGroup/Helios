<div align=center>
<img src="https://github.com/PKU-YuanGroup/Helios-Page/blob/main/figures/logo_white.png?raw=true" width="300px">
</div>

<h1 align="center">Helios: Real Real-Time Long Video Generation Model</h1>

<h5 align="center">⭐ 14B Real-Time Long Video Generation Model can be Cheaper, Faster but Keep Stronger than 1.3B ones ⭐</h5>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.04379-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2603.04379)
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2603.04379)
[![Project Page](https://img.shields.io/badge/Project-Website-2ea44f)](https://pku-yuangroup.github.io/Helios-Page)
[![hf_space](https://img.shields.io/badge/🤗-Gradio-00b4d8.svg)](https://huggingface.co/spaces/BestWishYsh/Helios-14B-RealTime)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-blue)](https://huggingface.co/collections/BestWishYsh/helios)
[![ModelScope](https://img.shields.io/badge/🤖-ModelScope-purple)](https://modelscope.cn/collections/BestWishYSH/Helios)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/PKU-YuanGroup/Helios)
[![GitCode](https://img.shields.io/badge/GitCodes-blue?logo=gitcode)](https://gitcode.com/weixin_47617277/Helios)

[![Ascend](https://img.shields.io/badge/Inference-Ascend--NPU-red)](https://www.hiascend.com/)
[![Diffusers](https://img.shields.io/badge/Inference-Diffusers-blueviolet)](https://github.com/huggingface/diffusers/pull/13208)
[![vLLM-Omni](https://img.shields.io/badge/Backend-vLLM--Omni-orange)](https://github.com/vllm-project/vllm-omni/pull/1604)
[![SGLang Diffusion](https://img.shields.io/badge/Backend-SGLang--Diffusion-yellow)](https://github.com/sgl-project/sglang/pull/19782)




</h5>

<div align="center">
This repository is the official implementation of Helios, which is a breakthrough video generation model that achieves minute-scale, high-quality video synthesis at <strong>19.5 FPS on a single H100 GPU</strong> (about 10 FPS on a single Ascend NPU) —without relying on conventional long video anti-drifting strategies or standard video acceleration techniques.
</div>

<br>

## ✨ Highlights


1. **Without commonly used anti-drifting strategies** (e.g., self-forcing, error-banks, keyframe sampling, or inverted sampling), Helios generates minute-scale videos with high quality and strong coherence.

2. **Without standard acceleration techniques** (e.g., KV-cache, causal masking, sparse/linear attention, TinyVAE, progressive noise schedules, hidden-state caching, or quantization), Helios achieves 19.5 FPS in end-to-end inference on a single H100 GPU.

3. **We introduce optimizations that improve both training and inference throughput while reducing memory consumption,** enabling image-diffusion-scale batch sizes during training while fitting up to four 14B models within 80 GB of GPU memory.



## 🎬 Video Demos

<!-- <div align="center">
  <video src="https://github.com/PKU-YuanGroup/Helios-Page/blob/main/videos/helios_features.mp4?raw=true" width="70%" controls="controls" poster=""></video>
</div>

or you can click <a href="https://www.youtube.com/watch?v=vd_AgHtOUFQ">here</a> to get the video. Some best prompts are [here](./example/prompt.txt). -->

[![Demo Video of Helios](https://github.com/user-attachments/assets/1d10da4a-aba9-4ac1-ab02-cd0dfce8d35b)](https://www.youtube.com/watch?v=vd_AgHtOUFQ)
or you can click <a href="https://github.com/PKU-YuanGroup/Helios-Page/blob/main/videos/helios_features.mp4">here</a> to get the video. Some best prompts are [here](./example/prompt.txt).


## 📣 Latest News!!

* `[2026.03.05]` 👋 We official release the [Gradio Demo](https://huggingface.co/spaces/BestWishYsh/Helios-14B-RealTime), welcome to try it.
* `[2026.03.04]` 👋 We are excited to announce the release of the Helios [technical report](https://arxiv.org/abs/2603.04379) on arXiv. We welcome discussions and feedback!
* `[2026.03.04]` 🚀 Day-0 support for [Ascend-NPU](https://www.hiascend.com)，with sincere gratitude to the Ascend Team for their support.
* `[2026.03.04]` 🚀 Day-0 support for [Diffusers](https://github.com/huggingface/diffusers/pull/13208)，with special thanks to the HuggingFace Team for their support.
* `[2026.03.04]` 🚀 Day-0 support for [vLLM-Omni](https://github.com/vllm-project/vllm-omni/pull/1604)，with heartfelt gratitude to the vLLM Team for their support.
* `[2026.03.04]` 🚀 Day-0 support for [SGLang-Diffusion](https://github.com/sgl-project/sglang/pull/19782)，with huge thanks to the SGLang Team for their support.
* `[2026.03.04]` 🔥 We've released the training/inference code and weights of **Helios-Base**, **Helios-Mid** and **Helios-Distilled**.


## 🔥 Friendly Links

If your work has improved **Helios** and you would like more people to see it, please inform us.

* [Ascend-NPU](https://www.hiascend.com/): Developed by Huawei, this hardware is designed for efficient AI model training and inference, boosting performance in tasks like computer vision, natural language processing, and autonomous driving.
* [Diffusers](https://github.com/huggingface/diffusers): A popular library designed for working with diffusion models and other generative models in deep learning. It supports easy integration and manipulation of a wide range of generative models.
* [vLLM-Omni](https://github.com/vllm-project/vllm-omni): A fully disaggregated serving system for any-to-any models. vLLM-Omni breaks complex architectures into a stage-based graph, using a decoupled backend to maximize resource efficiency and throughput.
* [SGLang-Diffusion](https://github.com/sgl-project/sglang): An inference framework for accelerated image and video generation using diffusion models. It provides an end-to-end unified pipeline with optimized kernels and an efficient scheduler loop.


## ⚙️ Requirements and Installation

### Prepare Environment

```bash
# 0. Clone the repo
git clone --depth=1 https://github.com/PKU-YuanGroup/Helios.git
cd Helios

# 1. Create conda environment
conda create -n helios python=3.11.2
conda activate helios

# 2. Install PyTorch (adjust for your CUDA version)
# CUDA 12.6
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# CUDA 13.0
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# 3. Install dependencies
bash install.sh
```

### Model Download

| Models           | Download Link                                                                                                                                            | Supports                                      | Notes                                                                                       |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------|
| Helios-Base      | 🤗 [Huggingface](https://huggingface.co/BestWishYsh/Helios-Base) 🤖 [ModelScope](https://modelscope.cn/models/BestWishYSH/Helios-Base)                | T2V ✅ I2V ✅ V2V ✅ Interactive ✅           | Best Quality, with v-prediction, standard CFG and custom HeliosScheduler.                   |
| Helios-Mid       | 🤗 [Huggingface](https://huggingface.co/BestWishYsh/Helios-Mid) 🤖 [ModelScope](https://modelscope.cn/models/BestWishYSH/Helios-Mid)                  | T2V ✅ I2V ✅ V2V ✅ Interactive ✅           | Intermediate Ckpt, with v-prediction, CFG-Zero* and custom HeliosScheduler.                 |
| Helios-Distilled | 🤗 [Huggingface](https://huggingface.co/BestWishYsh/Helios-Distilled) 🤖 [ModelScope](https://modelscope.cn/models/BestWishYSH/Helios-Distilled)      | T2V ✅ I2V ✅ V2V ✅ Interactive ✅           | Best Efficiency, with x0-prediction and custom HeliosDMDScheduler.                          |



> 💡Note: 
> * All three models share the same architecture, but Helios-Mid and Helios-Distilled use a more aggressive multi-scale sampling pipeline to achieve better efficiency.
> * Helios-Mid is an intermediate checkpoint generated in the process of distilling Helios-Base into Helios-Distilled, and may not meet expected quality.
> * For Image-to-Video or Video-to-Video, since training is based on Text-to-Video, these two functions may be slightly inferior to Text-to-Video. You may enable `is_skip_first_chunk` if you find the first few chunks are static.


Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download BestWishYSH/Helios-Base --local-dir BestWishYSH/Helios-Base
huggingface-cli download BestWishYSH/Helios-Mid --local-dir BestWishYSH/Helios-Mid
huggingface-cli download BestWishYSH/Helios-Distilled --local-dir BestWishYSH/HeliosDistillede
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download BestWishYSH/Helios-Base --local_dir BestWishYSH/Helios-Base
modelscope download BestWishYSH/Helios-Mid --local-dir BestWishYSH/Helios-Mid
modelscope download BestWishYSH/Helios-Distilled --local-dir BestWishYSH/HeliosDistillede
```

## 🚀 Inference


Helios uses an autoregressive approach that generates **33 frames per chunk**. For optimal performance, `num_frames` should be set to a multiple of `33`. If a non-multiple value is provided, it will be automatically rounded up to the nearest multiple of 33.

**Example frame counts for different video lengths:**

| num_frames | Adjusted Frames | 24 FPS | 16 FPS |
|------------|-----------------|--------|--------|
| 1449       | 1452 (33×44)    | ~60s (1min) | ~90s (1min 30s) |
| 720        | 726 (33×22)     | ~30s | ~45s |
| 240        | 264 (33×8)      | ~11s | ~16s |
| 129        | 132 (33×4)      | ~5.5s | ~8s |
| 81         | 99  (33×3)      | ~4s | ~6s |

### Run the model

We provide inference scripts for all models covering text-to-video, image-to-video, and video-to-video in this [directory](./scripts/inference).

```bash
cd scripts/inference

# For Helios-Base
bash helios-base_t2v.sh
bash helios-base_i2v.sh
bash helios-base_v2v.sh

# For Helios-Mid
bash helios-mid_t2v.sh
bash helios-mid_i2v.sh
bash helios-mid_v2v.sh

# For Helios-Distilled
bash helios-distilled_t2v.sh
bash helios-distilled_i2v.sh
bash helios-distilled_v2v.sh
```

### Sanity Check

Before trying your own inputs, we highly recommend going through the sanity check to find out if any hardware or software went wrong.

| Task    | **Helios-Base**                                                                                                            | **Helios-Mid**                                                                                                             | **Helios-Distilled**                                                                                                       |
| ------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **T2V** | <video src="https://github.com/user-attachments/assets/14e10753-0366-4790-ad8f-7b66d821ed11" controls width="240"></video> | <video src="https://github.com/user-attachments/assets/c1778691-a80b-428c-8094-88bb1dd1d52b" controls width="240"></video> | <video src="https://github.com/user-attachments/assets/4ca28c79-9dfa-49de-9c3a-f4c7b6c766cd" controls width="240"></video> |
| **V2V** | <video src="https://github.com/user-attachments/assets/420cb572-85c2-42d8-98d7-37b0bc24c844" controls width="240"></video> | <video src="https://github.com/user-attachments/assets/7d703fa6-dc1a-4138-a897-e58cfd9236d6" controls width="240"></video> | <video src="https://github.com/user-attachments/assets/45329c55-1a25-459c-bbf0-4e584ec5b23d" controls width="240"></video> |


### ✨ Diffusers Pipeline

Install diffusers from source:
```bash
pip install git+https://github.com/huggingface/diffusers.git
```

For example, let's take Helios-Distilled.

<details>
  <summary>Click to expand the code</summary>

  ```bash
  import torch
  from diffusers import ModularPipeline, ClassifierFreeGuidance
  from diffusers.utils import export_to_video, load_image, load_video

  mod_pipe = ModularPipeline.from_pretrained("BestWishYsh/Helios-Distilled")
  mod_pipe.load_components(torch_dtype=torch.bfloat16)
  mod_pipe.to("cuda")

  # we need to upload guider to the model repo, so each checkpoint will be able to config their guidance differently
  guider = ClassifierFreeGuidance(guidance_scale=1.0)
  mod_pipe.update_components(guider=guider)

  # --- T2V ---
  print("=== T2V ===")
  prompt = (
      "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. "
      "The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving "
      "fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and "
      "sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef "
      "itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures "
      "the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. "
      "A close-up shot with dynamic movement."
  )

  output = mod_pipe(
      prompt=prompt,
      height=384,
      width=640,
      num_frames=240,
      pyramid_num_inference_steps_list=[2, 2, 2],
      is_amplify_first_chunk=True,
      generator=torch.Generator("cuda").manual_seed(42),
      output="videos",
  )

  export_to_video(output[0], "helios_distilled_modular_t2v_output.mp4", fps=24)
  print(f"T2V max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

  # --- I2V ---
  print("=== I2V ===")
  image = load_image(
      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/wave.jpg"
  )
  i2v_prompt = (
      "A towering emerald wave surges forward, its crest curling with raw power and energy. "
      "Sunlight glints off the translucent water, illuminating the intricate textures and deep green hues within the wave's body."
  )

  output = mod_pipe(
      prompt=i2v_prompt,
      image=image,
      height=384,
      width=640,
      num_frames=240,
      pyramid_num_inference_steps_list=[2, 2, 2],
      is_amplify_first_chunk=True,
      generator=torch.Generator("cuda").manual_seed(42),
      output="videos",
  )

  export_to_video(output[0], "helios_distilled_modular_i2v_output.mp4", fps=24)
  print(f"I2V max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

  # --- V2V ---
  print("=== V2V ===")
  video = load_video(
      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/car.mp4"
  )
  v2v_prompt = (
      "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. "
      "The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, "
      "and distant mountain ranges passing by quickly."
  )

  output = mod_pipe(
      prompt=v2v_prompt,
      video=video,
      height=384,
      width=640,
      num_frames=240,
      pyramid_num_inference_steps_list=[2, 2, 2],
      is_amplify_first_chunk=True,
      generator=torch.Generator("cuda").manual_seed(42),
      output="videos",
  )

  export_to_video(output[0], "helios_distilled_modular_v2v_output.mp4", fps=24)
  print(f"V2V max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
  ```

</details>

### ✨ vLLM-Omni Pipeline

Install vllm-omni from source:
```bash
pip install git+https://github.com/vllm-project/vllm-omni.git
```

For example, let's take Text-to-Video.

<details>
  <summary>Click to expand the code</summary>

  ```bash
  cd vllm-omni

  # Helios-Base
  python3 examples/offline_inference/helios/end2end.py \
    --sample-type t2v \
    --model ./Helios-Base \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
    --num-frames 99 \
    --seed 42 \
    --output helios_t2v_base.mp4

  # Helios-Mid
  python examples/offline_inference/helios/end2end.py \
    --model ./Helios-Mid --sample-type t2v \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
    --guidance-scale 5.0 --is-enable-stage2 \
    --pyramid-num-inference-steps-list 20 20 20 \
    --num-frames 99 \
    --use-cfg-zero-star --use-zero-init --zero-steps 1 \
    --output helios_t2v_mid.mp4

  # Helios-Distilled
  python examples/offline_inference/helios/end2end.py \
    --model ./Helios-Distilled --sample-type t2v \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
    --num-frames 240 --guidance-scale 1.0 --is-enable-stage2 \
    --pyramid-num-inference-steps-list 2 2 2 \
    --is-amplify-first-chunk --output helios_t2v_distilled.mp4
  ```
</details>

### ✨ SGLang-Diffusion Pipeline

Install sglang-diffusion from source:
```bash
pip install git+https://github.com/sgl-project/sglang.git
```

For example, let's take Helios-Base. **(Native Support)**

<details>
  <summary>Click to expand the code</summary>

  ```bash
  sglang generate \
    --model-path BestWishYsh/Helios-Base \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --height 384 \
    --width 640 \
    --num-frames 99 \
    --num-inference-steps 50 \
    --guidance-scale 5.0
  ```
</details>

For example, let's take Helios-Base. **(Diffusers Backend)**

<details>
  <summary>Click to expand the code</summary>

  ```bash
  sglang generate \
    --model-path BestWishYsh/Helios-Base \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --height 384 \
    --width 640 \
    --num-frames 99 \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --backend diffusers
  ```
</details>

## 🗝️ Training

We use a three-stage progressive pipeline, all the setting can be found [here](./scripts/training/configs/). Stage-1 (Base) performs architectural adaptation: we apply Unified History Injection, Easy Anti-Drifting, and Multi-Term Memory Patchification to convert the bidirectional pretrained model into an autoregressive generator. Stage-2 (Mid) targets token compression by introducing Pyramid Unified Predictor Corrector, which aggressively reduces the number of noisy tokens and thus the overall computation. Stage-3 (Distilled) applies Adversarial Hierarchical Distillation, reducing the sampling steps from 50 to 3 and eliminating the need for classifier-free guidance (CFG). Throughout training, we apply dynamic shifting to all timestep-dependent operations to match the noise schedule to the latent size.

### Data Preparation

Please refer to [this guide](./tools/offload_data/README.md) for how to obtain the training data required by Helios. And we prepare a toy training data [here](https://huggingface.co/BestWishYsh/HeliosBench-Weights/tree/main/demo_data).

### Run the model

```bash
# Use DDP
bash scripts/training/train_ddp.sh

# or

# Use DeepSpeed
bash scripts/training/train_deepspeed.sh
```

Training configuration can be adjusted in `scripts/training/configs`. You can use `scripts/training/compare_yaml.py` to check for configuration completeness or differences between stages.


## 📊 HeliosBench

HeliosBench is a specialized benchmark for real-time long-video generation, please refer to [this guide](./eval/README.md) for how to eval your own model.


## 👍 Acknowledgement

This project wouldn't be possible without the following open-sourced repositories: [Open-Sora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [Ascend](https://www.hiascend.com), [Diffusers](https://github.com/huggingface/diffusers), [vLLM-Omni](https://github.com/vllm-project/vllm), [SGLang Diffusion](https://github.com/sgl-project/sglang), [Wan](https://github.com/Wan-Video/Wan2.1), [FramePack](https://github.com/lllyasviel/FramePack), [PyramidFlow](https://github.com/jy0205/Pyramid-Flow), [DMD](https://github.com/tianweiy/DMD2).


## 🔒 License

This project is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝:

```BibTeX
@article{helios,
  title={Helios: Real Real-Time Long Video Generation Model},
  author={Yuan, Shenghai and Yin, Yuanyang and Li, Zongjian and Huang, Xinwei and Yang, Xiao and Yuan, Li},
  journal={arXiv preprint arXiv:2603.04379},
  year={2026}
}
```

## 🤝 Contact

For questions and feedback, please contact us at: shyuan-cs@hotmail.com


