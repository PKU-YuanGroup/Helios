from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReportTo:
    tracker_name: str = field(default="Spark-Wan")
    wandb_name: str = field(default="test_run")
    report_to: str = field(
        default="wandb",
        metadata={"choices": ["wandb", "tensorboard", "comet_ml", "all"]},
    )


@dataclass
class DataConfig:
    # ---- Base ----
    use_shuffle: bool = field(default=False)
    pin_memory: bool = field(default=False)
    persistent_workers: bool = field(default=False)
    instance_data_root: list = field(default_factory=list)
    instance_video_root: list = field(default_factory=list)
    dataset_sampling_ratios: list = field(default_factory=list)
    dataloader_num_workers: int = field(default=0)
    prefetch_factor: int = field(default=2)
    force_rebuild: bool = field(default=False)
    stride: int = field(default=1)
    resolution: int = field(default=640)
    single_res: bool = field(default=False)
    single_res: bool = field(default=False)
    single_height: int = field(default=384)
    single_width: int = field(default=640)
    single_length: bool = field(default=False)
    single_num_frame: int = field(default=81)
    multi_res: bool = field(default=False)
    caption_dropout_p: float = field(default=0.00)
    id_token: str = field(default="")
    negative_prompt: str = field(
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )
    # ---- Stage 1 ----
    use_stage1_dataset: bool = field(default=False)
    # ---- Stage 3 ----
    use_stage3_dataset: bool = field(default=False)
    gan_data_root: Optional[list] = field(default_factory=list)
    ode_data_root: Optional[list] = field(default_factory=list)
    text_data_root: Optional[list] = field(default_factory=list)


@dataclass
class ModelConfig:
    # ---- Path ----
    pretrained_model_name_or_path: Optional[str] = field(default=None)
    transformer_model_name_or_path: Optional[str] = field(default=None)
    siglip_model_name_or_path: Optional[str] = field(default=None)
    lora_paths: Optional[list[str]] = field(default_factory=list)
    subfolder: Optional[str] = field(default=None)
    revision: Optional[str] = field(default=None)
    variant: Optional[str] = field(default=None)
    load_checkpoints_custom: bool = field(default=False)
    load_model_path: Optional[str] = field(default=None)
    load_dcp: bool = field(default=False)
    load_dcp_path: Optional[str] = field(default=None)
    # ---- Vae ----
    upcast_vae: bool = field(default=False)
    enable_slicing: bool = field(default=False)
    enable_tiling: bool = field(default=False)
    # ---- Lora ----
    lora_rank: int = field(default=128)
    lora_alpha: float = field(default=128.0)
    lora_dropout: float = field(default=0.0)
    lora_layers: Optional[str] = field(default=None)
    lora_target_modules: list = field(default_factory=list)
    lora_exclude_modules: list = field(default_factory=list)
    # ---- Other ----
    train_norm_layers: bool = field(default=False)
    bnb_quantization_config_path: Optional[str] = field(default=None)
    # ----- Stage 3 -----
    critic_lora_name_or_path: Optional[str] = field(default=None)
    critic_subfolder: Optional[str] = field(default=None)
    critic_lora_rank: int = field(default=128)
    critic_lora_alpha: float = field(default=128.0)
    critic_lora_dropout: float = field(default=0.0)
    real_score_model_name_or_path: Optional[str] = field(default=None)
    # ---- Reward Parameters ----
    reward_model_name_or_path: Optional[str] = field(default=None)


@dataclass
class ValidationConfig:
    validation_steps: int = field(default=100)
    validation_height: int = field(default=480)
    validation_width: int = field(default=832)
    validation_max_num_frames: int = field(default=81)
    validation_prompts: Optional[list[str]] = field(default_factory=lambda: ["A frog jumps on a lotus leaf."])
    validation_images: Optional[list[str]] = field(default_factory=lambda: ["example/input_images/frog.jpg"])
    validation_guidance_scale: float = field(default=9.0)
    validation_latent_window_size: list[int] = field(default_factory=lambda: [9])
    validation_stream_chunk_size: list[int] = field(default_factory=lambda: [3])
    first_step_valid: bool = field(default=True)
    num_validation_videos: int = field(default=1)
    num_inference_steps: int = field(default=30)
    # ---- Stage 1 ----
    use_kv_cache: bool = field(default=False)
    # ---- Stage 2 ----
    stage2_simulated_inference_steps: list[int] = field(default_factory=lambda: [10, 10, 10])


@dataclass
class TrainingConfig:
    # ---- Environment ----
    local_rank: int = field(default=-1)
    allow_tf32: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    enable_xformers_memory_efficient_attention: bool = field(default=False)
    enable_npu_flash_attention: bool = field(default=False)
    upcast_before_saving: bool = field(default=False)
    offload: bool = field(default=False)
    mixed_precision: str = field(
        default="bf16",
        metadata={"choices": ["no", "fp16", "bf16"]},
    )
    profile_out_dir: Optional[str] = field(default=None)
    # ---- Training Resource ----
    num_train_epochs: int = field(default=1)
    max_train_steps: Optional[int] = field(default=None)
    train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    checkpointing_steps: int = field(default=500)
    checkpoints_total_limit: Optional[int] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_checkpoints_custom: bool = field(default=False)
    # ---- Optimizer ----
    learning_rate: float = field(default=2e-4)
    scale_lr: bool = field(default=False)
    lr_scheduler: str = field(
        default="constant",
        metadata={
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ]
        },
    )
    lr_warmup_steps: int = field(default=500)
    lr_num_cycles: int = field(default=1)
    lr_power: float = field(default=1.0)
    optimizer: str = field(
        default="adamw",
        metadata={
            "choices": ["adam", "adamw", "prodigy"],
        },
    )
    use_8bit_adam: bool = field(default=False)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    prodigy_beta3: Optional[float] = field(default=None)
    prodigy_decouple: bool = field(default=True)
    prodigy_use_bias_correction: bool = field(default=True)
    prodigy_safeguard_warmup: bool = field(default=True)
    adam_weight_decay: float = field(default=1e-04)
    adam_epsilon: float = field(default=1e-08)
    max_grad_norm: float = field(default=1.0)
    weighting_scheme: str = field(
        default="logit_normal",
        metadata={
            "choices": ["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        },
    )
    logit_mean: float = field(default=0.0)
    logit_std: float = field(default=1.0)
    mode_scale: float = field(default=1.29)
    # ---- Dynamic Shifting ----
    use_dynamic_shifting: bool = field(default=False)
    base_seq_len: Optional[int] = field(default=256)
    max_seq_len: Optional[int] = field(default=4096)
    base_shift: Optional[float] = field(default=0.5)
    max_shift: Optional[float] = field(default=1.15)
    # ---- VAE Decode Parameters ----
    vae_decode_type: str = field(
        default="default",
        metadata={
            "choices": ["default", "dafault_batch"],
        },
    )
    # ---- EMA ----
    use_ema: bool = field(default=False)
    use_ema_validation: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    ema_start_step: int = field(default=0)
    ema_zero3_port: int = field(default=10543)
    ema_deepspeed_config_file: str = field(default="scripts/accelerate_configs/zero3.json")
    # ---- Stage 1 Parameters ----
    is_enable_stage1: bool = field(default=False)
    history_sizes: list[int] = field(default_factory=lambda: [16, 2, 1])
    latent_window_size: list[int] = field(default_factory=lambda: [9])
    is_random_drop: bool = field(default=False)
    random_drop_i2v_ratio: float = field(default=0)
    random_drop_v2v_ratio: float = field(default=0)
    random_drop_t2v_ratio: float = field(default=0)
    is_amplify_history: bool = field(default=False)
    history_scale_mode: str = field(
        default="per_head",
        metadata={
            "choices": ["scalar", "per_head"],
        },
    )
    #
    has_multi_term_memory_patch: bool = field(default=False)
    is_train_full_multi_term_memory_patchg: bool = field(default=False)
    is_train_lora_multi_term_memory_patchg: bool = field(default=False)
    is_train_full_patch_embedding: bool = field(default=False)
    is_train_lora_patch_embedding: bool = field(default=False)
    zero_history_timestep: bool = field(default=False)
    restrict_self_attn: bool = field(default=False)
    guidance_cross_attn: bool = field(default=False)
    is_train_restrict_lora: bool = field(default=False)
    restrict_lora: bool = field(default=False)
    restrict_lora_rank: int = field(default=128)
    # ---- Easy Anti-Drifting Parameters ----
    corrupt_model_input: bool = field(default=False)
    corrupt_mode_model_input: str = field(
        default="noise",
        metadata={
            "choices": ["noise", "downsample", "random"],
        },
    )
    corrupt_mode_prob_model_input: float = field(default=0.9)
    is_frame_independent_corrupt_model_input: bool = field(default=False)
    is_chunk_independent_corrupt_model_input: bool = field(default=False)
    noise_corrupt_ratio_model_input: float = field(default=1 / 3)
    noise_corrupt_clean_prob_model_input: float = field(default=0.1)
    downsample_min_corrupt_ratio_model_input: float = field(default=0.9)
    downsample_max_corrupt_ratio_model_input: float = field(default=1.0)
    #
    corrupt_history: bool = field(default=False)
    corrupt_mode_history: str = field(
        default="noise",
        metadata={
            "choices": ["noise", "downsample", "random"],
        },
    )
    corrupt_mode_prob_history: float = field(default=0.9)
    is_frame_independent_corrupt_history: bool = field(default=False)
    is_chunk_independent_corrupt_history: bool = field(default=False)
    noise_corrupt_ratio_history_short: float = field(default=1 / 3)
    noise_corrupt_ratio_history_mid: float = field(default=1 / 3)
    noise_corrupt_ratio_history_long: float = field(default=1 / 3)
    noise_corrupt_clean_prob_history: float = field(default=0.1)
    downsample_min_corrupt_ratio_history: float = field(default=0.9)
    downsample_max_corrupt_ratio_history: float = field(default=1.0)
    #
    is_add_saturation: bool = field(default=False)
    saturation_ratio_min: float = field(default=0.3)
    saturation_ratio_max: float = field(default=1.7)
    saturation_ratio_clean_prob: float = field(default=0.1)
    # ---- Stage 2 Parameters ----
    is_enable_stage2: bool = field(default=False)
    is_navit_pyramid: bool = field(default=False)
    stage2_num_stages: int = field(default=3)
    stage2_timestep_shift: float = field(default=1.0)
    stage2_scheduler_gamma: float = field(default=1 / 3)
    stage2_stage_range: list[float] = field(default_factory=lambda: [0.0, 1 / 3, 2 / 3, 1])
    stage2_sample_ratios: list[int] = field(default_factory=lambda: [1, 2, 1])
    efficient_sample: bool = field(default=False)
    # ---- Stage 3 VRAM Parameters ----
    dmd_is_low_vram_mode: bool = field(default=False)
    is_gan_low_vram_mode: bool = field(default=False)
    dmd_is_offload_grad: bool = field(default=False)
    # ---- Stage 3 Parameters ----
    log_iters: int = field(default=200)
    no_visualize: bool = field(default=False)
    is_train_dmd: bool = field(default=False)
    max_grad_norm_critic: float = field(default=1.0)
    dmd_generator_deepspeed_config: Optional[str] = field(default=None)
    dmd_critic_deepspeed_config: Optional[str] = field(default=None)
    critic_learning_rate: Optional[float] = field(default=2e-6)
    dfake_gen_update_ratio: Optional[int] = field(default=5)
    dmd_denoising_step_list: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    num_critic_input_frames: Optional[int] = field(default=21)
    dmd_timestep_shift: Optional[float] = field(default=5.0)
    dmd_last_step_only: bool = field(default=False)
    dmd_last_section_grad_only: bool = field(default=False)
    dmd_teacher_forcing: bool = field(default=False)
    dmd_teacher_forcing_ratio: float = field(default=0.2)
    fake_guidance_scale: float = field(default=0.0)
    real_guidance_scale: float = field(default=3.0)
    is_skip_first_section: bool = field(default=False)
    is_amplify_first_chunk: bool = field(default=False)
    # ---- GT History Parameters ----
    is_use_gt_history: bool = field(default=False)
    use_gt_history_ratio: float = field(default=1.0)
    is_use_gt_coherence_dmd: bool = field(default=False)
    # ---- VAE Re-Encode ----
    is_dmd_vae_decode: bool = field(default=False)
    # ---- Multi Stage Backward Simulated ----
    is_multi_pyramid_stage_backward_simulated: bool = field(default=False)
    # ---- Consistency Align Parameters ----
    is_consistency_align: bool = field(default=False)
    consistentcy_align_weight: float = field(default=0.25)
    # ---- Smoothness Parameters ----
    is_smoothness_loss: bool = field(default=False)
    smoothness_loss_weight: float = field(default=1e-2)
    # ---- Mean-Variance Regularization Parameters ----
    is_mean_var_regular: bool = field(default=False)
    mean_var_regular_weight: float = field(default=1.0)
    regular_mean: Optional[float] = field(default=0.00657021)
    regular_var: Optional[float] = field(default=0.85126512)
    is_x0_mean_var_regular: bool = field(default=False)
    mean_var_regular_x0_weight: float = field(default=1.0)
    regular_x0_mean: Optional[float] = field(default=-0.01618061)
    regular_x0_var: Optional[float] = field(default=0.27996052)
    #
    is_chunk_mean_var_regular: bool = field(default=False)
    chunk_mean_var_regular_weight: float = field(default=1.0)
    chunk_regular_mean: Optional[float] = field(default=0.01906107)
    chunk_regular_var: Optional[float] = field(default=0.81397036)
    is_chunk_x0_mean_var_regular: bool = field(default=False)
    chunk_mean_var_regular_x0_weight: float = field(default=1.0)
    chunk_regular_x0_mean: Optional[float] = field(default=-0.01578601)
    chunk_regular_x0_var: Optional[float] = field(default=0.29913200)
    # ---- ODE Regression ----
    is_use_ode_regression: bool = field(default=False)
    is_only_ode_regression: bool = field(default=False)
    ode_regression_weight: float = field(default=0.25)
    ode_num_latent_sections_min: int = field(default=3)
    ode_num_latent_sections_max: int = field(default=3)
    # ---- GAN Parameters ----
    is_use_gan: bool = field(default=False)
    gan_start_step: int = field(default=0)
    is_separate_gan_grad: bool = field(default=False)
    is_use_gan_hooks: bool = field(default=False)
    is_use_gan_final: bool = field(default=False)
    gan_cond_map_dim: int = field(default=768)
    gan_hooks: list[int] = field(default_factory=lambda: [5, 15, 25, 35])
    gan_g_weight: float = field(default=1e-2)
    gan_d_weight: float = field(default=1e-2)
    aprox_r1: bool = field(default=False)
    aprox_r2: bool = field(default=False)
    r1_weight: float = field(default=0.0)
    r2_weight: float = field(default=0.0)
    r1_sigma: float = field(default=0.1)
    r2_sigma: float = field(default=0.1)
    # ---- Reward Parameters ----
    is_use_reward_model: bool = field(default=False)
    reward_start_step: int = field(default=0)
    reward_weight_vq: float = field(default=2.0)
    reward_weight_mq: float = field(default=2.0)
    reward_weight_ta: float = field(default=2.0)
    # ---- Decouple Parameters ----
    is_decouple_dmd: bool = field(default=False)
    decouple_ca_start_step: int = field(default=2000)
    decouple_ca_end_step: int = field(default=3000)
    # ---- Cold Start Parameters ----
    is_enable_cold_start: bool = field(default=False)
    cold_start_step: int = field(default=1000)
    stage_cold_start_step: Optional[int] = field(default=None)
    # ---- Dynamic Timestep ----
    generator_is_forcing_low_renoise: bool = field(default=False)
    generator_dynamic_alpha: float = field(default=4.0)
    generator_dynamic_beta: float = field(default=1.5)
    generator_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    generator_dynamic_step: int = field(default=1000)
    critic_dynamic_alpha: float = field(default=4.0)
    critic_dynamic_beta: float = field(default=1.5)
    critic_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    critic_dynamic_step: int = field(default=1000)
    # ---- Dynamic DMD Section ----
    dmd_num_latent_sections_min: Optional[int] = field(default=3)
    dmd_num_latent_sections_max: Optional[int] = field(default=3)
    dmd_dynamic_alpha: float = field(default=1.5)
    dmd_dynamic_beta: float = field(default=4.0)
    dmd_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    dmd_dynamic_step: int = field(default=1000)
    # ---- Dynamic ODE Section ----
    ode_dynamic_alpha: float = field(default=1.5)
    ode_dynamic_beta: float = field(default=4.0)
    ode_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    ode_dynamic_step: int = field(default=1000)
    # ---- Recycle ----
    use_error_recycling: bool = field(default=False)
    y_error_sample_from_all_grids: bool = field(default=True)

    error_buffer_size: int = field(default=500)
    buffer_replacement_strategy: str = field(default="l2_batch")
    buffer_warmup_iter: int = field(default=50)
    timestep_grid_size: int = field(default=25)
    num_grids: int = field(default=50)

    y_error_num: int = field(default=6)
    error_modulate_factor: float = field(default=0.0)
    error_setting: int = field(default=1)
    noise_prob: float = field(default=0.01)
    y_prob: float = field(default=0.9)
    latent_prob: float = field(default=0.9)
    clean_prob: float = field(default=0.2)
    clean_buffer_update_prob: float = field(default=0.1)


@dataclass
class Args:
    output_dir: str = field(default="Helios")
    seed: int = field(default=42)
    report_to: ReportTo = field(default_factory=ReportTo)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    logging_dir: str = field(default="logs")
