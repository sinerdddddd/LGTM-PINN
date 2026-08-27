from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = Path("TimeSeriesData_2")
DEFAULT_PARAMETER_PATH = Path("Parameter")


def _project_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


@dataclass
class LGTMConfig:
    data_path: Path = DEFAULT_DATA_PATH
    csv_file: Optional[Path] = None
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")
    parameter_raster_dir: Path = DEFAULT_PARAMETER_PATH

    use_csv_input: bool = False
    file_format: Optional[str] = "tif"
    raster_variable: Optional[str] = None
    raster_band: int = 1
    resize_shape: Optional[Tuple[int, int]] = None
    apply_scale_offset: bool = True
    invalid_fill_value: float = 0.0
    unreadable_policy: str = "nearest"
    max_fallback_search: Optional[int] = None
    unreadable_report_limit: int = 20
    validate_files: bool = False
    cache_size: int = 0
    stats_sample_limit: Optional[int] = None

    # Time split uses chronological scenes: train, test, then validation.
    seq_length: int = 10
    pred_length: int = 1
    train_timesteps: int = 100
    test_timesteps: int = 20
    val_timesteps: int = 16

    # Kept for backward-compatible config snapshots; temporal splitting
    # uses full spatial rasters for every split.
    random_seed: int = 20260612
    train_ratio: float = 0.70
    test_ratio: float = 0.15
    val_ratio: float = 0.0

    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    num_epochs: int = 150
    patience: int = 15
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    checkpoint_interval: int = 20
    test_batches: Optional[int] = None
    gradient_clip: float = 1.0
    batch_cleanup_interval: int = 50
    epoch_cleanup_interval: int = 1
    clear_reader_cache_each_epoch: bool = True
    save_optimizer_state: bool = False
    checkpoint_on_cpu: bool = True
    loss_history_limit: int = 1000

    # Legacy patch settings are kept only for old config snapshots and CLI
    # compatibility. The current model consumes the whole raster at once.
    patch_size: Tuple[int, int] = (64, 64)
    patch_stride: Tuple[int, int] = (32, 32)
    patch_stage_size: int = 4
    num_channels: int = 1
    informer_d_model: int = 64
    informer_n_heads: int = 4
    informer_e_layers: int = 2
    informer_d_ff: int = 128
    informer_dropout: float = 0.1
    unet_features: Tuple[int, ...] = (32, 64, 128, 256)
    use_global_refiner: bool = False
    global_refiner_features: Tuple[int, ...] = (16, 32)
    residual_prediction: bool = False

    loss_name: str = "huber"
    huber_delta: float = 1.0
    data_loss_weight: float = 1.0
    residual_loss_weight: float = 0.0
    residual_gradient_loss_weight: float = 0.0
    residual_distribution_loss_weight: float = 0.0
    residual_scale_min: float = 1e-3
    use_pinn_loss: bool = True
    pinn_loss_weight: float = 1e-6
    use_adaptive_pinn_weighting: bool = True
    adaptive_pinn_initial_weight: float = 1e-6
    adaptive_pinn_target_ratio: float = 0.2
    adaptive_pinn_ema_beta: float = 0.8
    adaptive_pinn_min_weight: float = 1e-8
    adaptive_pinn_max_weight: float = 2e-3
    adaptive_pinn_epsilon: float = 1e-12
    enable_gradient_conflict_diagnostics: bool = True
    gradient_diagnostics_interval: int = 1
    gradient_diagnostics_max_batches: int = 1
    gradient_diagnostics_split: str = "train"
    gradient_diagnostics_filename: str = "gradient_conflict_diagnostics.csv"
    use_gradient_angle_pinn_control: bool = True
    gradient_angle_control_warmup_epochs: int = 5
    gradient_angle_cosine_ema_beta: float = 0.8
    gradient_gate_positive_cosine: float = 0.3
    gradient_gate_negative_cosine: float = -0.3
    gradient_gate_neutral_value: float = 0.5
    gradient_gate_min_value: float = 0.05
    report_raw_pinn_loss: bool = True
    load_pinn_parameters: bool = True
    pinn_mode: str = "soft3_dominant"
    time_step: float = 1.0
    depth_step: float = 5.0
    cv_is_log10: bool = False
    cv_cm2s_to_m2day: bool = False
    cv_cm2year_to_m2day: bool = True
    settlement_sign: float = -1.0
    pinn_residual_scale_mm_per_day: float = 1.0
    soft3_drainage: str = "single"
    stress_scale_kpa: float = 1.0
    parameter_nodata_fill: float = 0.0
    required_parameter_names: Tuple[str, ...] = (
        "cv_mean",
        "Sinf_sum_mm",
        "contribution_weight_soft_only",
        "Hdr_single_m",
        "Hdr_double_m",
    )

    prediction_suffix: str = "lgtm_lager_pinn"
    save_predictions: bool = True
    future_steps: int = 50
    metric_iou_threshold: float = 0.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.data_path = _project_path(Path(self.data_path))
        self.csv_file = _project_path(Path(self.csv_file)) if self.csv_file is not None else None
        self.output_dir = _project_path(Path(self.output_dir))
        self.model_dir = _project_path(Path(self.model_dir))
        self.parameter_raster_dir = _project_path(Path(self.parameter_raster_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self.seq_length < 1:
            raise ValueError("seq_length must be >= 1")
        if self.pred_length < 1:
            raise ValueError("pred_length must be >= 1")
        if self.train_timesteps < self.seq_length + self.pred_length:
            raise ValueError("train_timesteps must allow at least one complete sliding window")
        if self.pred_length != 1:
            raise ValueError("The current LGTM_PINN model is trained as a one-step predictor; keep pred_length=1.")
        if self.test_timesteps < 1:
            raise ValueError("test_timesteps must be >= 1")
        if self.val_timesteps < 1:
            raise ValueError("val_timesteps must be >= 1")
        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0, 1)")
        if not (0.0 <= self.test_ratio < 1.0):
            raise ValueError("test_ratio must be in [0, 1)")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError("val_ratio must be in [0, 1)")
        if self.patch_stage_size < 1:
            raise ValueError("patch_stage_size must be >= 1")
        if len(self.patch_stride) != 2 or self.patch_stride[0] < 1 or self.patch_stride[1] < 1:
            raise ValueError("patch_stride must contain two positive integers")
        if self.patch_stride[0] > self.patch_size[0] or self.patch_stride[1] > self.patch_size[1]:
            raise ValueError("patch_stride must be <= patch_size in both directions")
        if self.unreadable_policy not in {"nearest", "zeros", "raise"}:
            raise ValueError("unreadable_policy must be one of: nearest, zeros, raise")
        if self.max_fallback_search is not None and self.max_fallback_search < 1:
            raise ValueError("max_fallback_search must be >= 1 or None")
        if self.unreadable_report_limit < 0:
            raise ValueError("unreadable_report_limit must be >= 0")
        if self.batch_cleanup_interval < 0:
            raise ValueError("batch_cleanup_interval must be >= 0")
        if self.epoch_cleanup_interval < 0:
            raise ValueError("epoch_cleanup_interval must be >= 0")
        if self.loss_history_limit < 1:
            raise ValueError("loss_history_limit must be >= 1")
        if self.future_steps < 0:
            raise ValueError("future_steps must be >= 0")
        if self.pinn_mode not in {"auto", "terzaghi_1d", "depth_integrated", "soft3_dominant"}:
            raise ValueError("pinn_mode must be one of: auto, terzaghi_1d, depth_integrated, soft3_dominant")
        if self.data_loss_weight < 0:
            raise ValueError("data_loss_weight must be >= 0")
        if self.residual_loss_weight < 0:
            raise ValueError("residual_loss_weight must be >= 0")
        if self.residual_gradient_loss_weight < 0:
            raise ValueError("residual_gradient_loss_weight must be >= 0")
        if self.residual_distribution_loss_weight < 0:
            raise ValueError("residual_distribution_loss_weight must be >= 0")
        if self.residual_scale_min <= 0:
            raise ValueError("residual_scale_min must be > 0")
        if self.pinn_loss_weight < 0:
            raise ValueError("pinn_loss_weight must be >= 0")
        if self.adaptive_pinn_initial_weight < 0:
            raise ValueError("adaptive_pinn_initial_weight must be >= 0")
        if self.adaptive_pinn_target_ratio < 0:
            raise ValueError("adaptive_pinn_target_ratio must be >= 0")
        if not (0.0 <= self.adaptive_pinn_ema_beta < 1.0):
            raise ValueError("adaptive_pinn_ema_beta must be in [0, 1)")
        if self.adaptive_pinn_min_weight < 0:
            raise ValueError("adaptive_pinn_min_weight must be >= 0")
        if self.adaptive_pinn_max_weight < self.adaptive_pinn_min_weight:
            raise ValueError("adaptive_pinn_max_weight must be >= adaptive_pinn_min_weight")
        if self.adaptive_pinn_epsilon <= 0:
            raise ValueError("adaptive_pinn_epsilon must be > 0")
        if self.gradient_diagnostics_interval < 1:
            raise ValueError("gradient_diagnostics_interval must be >= 1")
        if self.gradient_diagnostics_max_batches < 1:
            raise ValueError("gradient_diagnostics_max_batches must be >= 1")
        if self.gradient_diagnostics_split not in {"train", "test", "val", "all"}:
            raise ValueError("gradient_diagnostics_split must be one of: train, test, val, all")
        if not self.gradient_diagnostics_filename:
            raise ValueError("gradient_diagnostics_filename must be non-empty")
        if self.gradient_angle_control_warmup_epochs < 0:
            raise ValueError("gradient_angle_control_warmup_epochs must be >= 0")
        if not (0.0 <= self.gradient_angle_cosine_ema_beta < 1.0):
            raise ValueError("gradient_angle_cosine_ema_beta must be in [0, 1)")
        if not (0.0 < self.gradient_gate_positive_cosine <= 1.0):
            raise ValueError("gradient_gate_positive_cosine must be in (0, 1]")
        if not (-1.0 <= self.gradient_gate_negative_cosine < 0.0):
            raise ValueError("gradient_gate_negative_cosine must be in [-1, 0)")
        if not (0.0 <= self.gradient_gate_min_value <= 1.0):
            raise ValueError("gradient_gate_min_value must be in [0, 1]")
        if not (self.gradient_gate_min_value <= self.gradient_gate_neutral_value <= 1.0):
            raise ValueError("gradient_gate_neutral_value must be between gradient_gate_min_value and 1")
        if self.time_step <= 0:
            raise ValueError("time_step must be > 0")
        if self.depth_step <= 0:
            raise ValueError("depth_step must be > 0")
        if self.settlement_sign not in {-1.0, 1.0}:
            raise ValueError("settlement_sign must be -1.0 or 1.0")
        if self.pinn_residual_scale_mm_per_day <= 0:
            raise ValueError("pinn_residual_scale_mm_per_day must be > 0")
        if self.soft3_drainage not in {"single", "double"}:
            raise ValueError("soft3_drainage must be one of: single, double")
        if self.lr_scheduler_patience < 1:
            raise ValueError("lr_scheduler_patience must be >= 1")
        if not (0.0 < self.lr_scheduler_factor < 1.0):
            raise ValueError("lr_scheduler_factor must be in (0, 1)")
        if self.use_csv_input and self.csv_file is None:
            raise ValueError("csv_file is required when use_csv_input=True")
        if self.num_workers == 0:
            self.persistent_workers = False
            self.prefetch_factor = None
