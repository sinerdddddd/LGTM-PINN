import argparse
import csv
import gc
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .config import LGTMConfig
    from .data_input import LGTMDataModule
    from .losses import build_loss
    from .models import PatchwiseRasterForecastModel
except ImportError:
    from config import LGTMConfig
    from data_input import LGTMDataModule
    from losses import build_loss
    from models import PatchwiseRasterForecastModel


EXPLICIT_LGTM_CONFIG = {
    "data_path": Path("TimeSeriesData_2"),
    "csv_file": None,
    "output_dir": Path("outputs"),
    "model_dir": Path("models"),
    "parameter_raster_dir": Path("Parameter"),
    "use_csv_input": False,
    "file_format": "tif",
    "raster_variable": None,
    "raster_band": 1,
    "resize_shape": None,
    "apply_scale_offset": True,
    "invalid_fill_value": 0.0,
    "unreadable_policy": "nearest",
    "max_fallback_search": None,
    "unreadable_report_limit": 20,
    "validate_files": False,
    "cache_size": 0,
    "stats_sample_limit": None,
    "seq_length": 10,
    "pred_length": 1,
    "train_timesteps": 100,
    "test_timesteps": 20,
    "val_timesteps": 16,
    "random_seed": 20260612,
    "train_ratio": 0.70,
    "test_ratio": 0.15,
    "val_ratio": 0.0,
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
    "prefetch_factor": None,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,
    "num_epochs": 150,
    "patience": 15,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
    "checkpoint_interval": 20,
    "test_batches": None,
    "gradient_clip": 1.0,
    "batch_cleanup_interval": 50,
    "epoch_cleanup_interval": 1,
    "clear_reader_cache_each_epoch": True,
    "save_optimizer_state": False,
    "checkpoint_on_cpu": True,
    "loss_history_limit": 1000,
    "patch_size": (64, 64),
    "patch_stride": (32, 32),
    "patch_stage_size": 4,
    "num_channels": 1,
    "informer_d_model": 64,
    "informer_n_heads": 4,
    "informer_e_layers": 2,
    "informer_d_ff": 128,
    "informer_dropout": 0.1,
    "unet_features": (32, 64, 128, 256),
    "use_global_refiner": False,
    "global_refiner_features": (16, 32),
    "residual_prediction": False,
    "loss_name": "huber",
    "huber_delta": 1.0,
    "data_loss_weight": 1.0,
    "residual_loss_weight": 0.0,
    "residual_gradient_loss_weight": 0.0,
    "residual_distribution_loss_weight": 0.0,
    "residual_scale_min": 1e-3,
    "use_pinn_loss": True,
    "pinn_loss_weight": 1e-6,
    "use_adaptive_pinn_weighting": True,
    "adaptive_pinn_initial_weight": 1e-6,
    "adaptive_pinn_target_ratio": 0.2,
    "adaptive_pinn_ema_beta": 0.8,
    "adaptive_pinn_min_weight": 1e-8,
    "adaptive_pinn_max_weight": 2e-3,
    "adaptive_pinn_epsilon": 1e-12,
    "enable_gradient_conflict_diagnostics": True,
    "gradient_diagnostics_interval": 1,
    "gradient_diagnostics_max_batches": 1,
    "gradient_diagnostics_split": "train",
    "gradient_diagnostics_filename": "gradient_conflict_diagnostics.csv",
    "use_gradient_angle_pinn_control": True,
    "gradient_angle_control_warmup_epochs": 5,
    "gradient_angle_cosine_ema_beta": 0.8,
    "gradient_gate_positive_cosine": 0.3,
    "gradient_gate_negative_cosine": -0.3,
    "gradient_gate_neutral_value": 0.5,
    "gradient_gate_min_value": 0.05,
    "report_raw_pinn_loss": True,
    "load_pinn_parameters": True,
    "pinn_mode": "soft3_dominant",
    "time_step": 1.0,
    "depth_step": 5.0,
    "cv_is_log10": False,
    "cv_cm2s_to_m2day": False,
    "cv_cm2year_to_m2day": True,
    "settlement_sign": -1.0,
    "pinn_residual_scale_mm_per_day": 1.0,
    "soft3_drainage": "single",
    "stress_scale_kpa": 1.0,
    "parameter_nodata_fill": 0.0,
    "required_parameter_names": (
        "cv_mean",
        "Sinf_sum_mm",
        "contribution_weight_soft_only",
        "Hdr_single_m",
        "Hdr_double_m",
    ),
    "prediction_suffix": "lgtm_lager_pinn_gradgate",
    "save_predictions": True,
    "future_steps": 50,
    "metric_iou_threshold": 0.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


USE_PRETRAINED_BEST_MODEL = False
STRICT_PRETRAINED_LOAD = True
RESUME_OPTIMIZER_STATE = False
KEEP_BEST_LOSS_FROM_CHECKPOINT = False


class LGTM:
    def __init__(self, config: Optional[LGTMConfig] = None) -> None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
        self.config = config or LGTMConfig()
        self.device = torch.device(self.config.device)
        self.data: Optional[LGTMDataModule] = None
        self.model: Optional[PatchwiseRasterForecastModel] = None

    def setup_data(self) -> LGTMDataModule:
        self.data = LGTMDataModule(self.config).setup()
        summary = self.data.summary()
        normalizer = summary["normalizer"]
        print("=" * 80)
        print("LGTM + PINN data input")
        print(f"Data path: {self.config.data_path}")
        print(f"Input order: 8-digit tif filename ascending")
        print(f"Timesteps: {summary['timesteps']} ({summary['first_file']} -> {summary['last_file']})")
        print(
            f"Task: one-step LGTM training with {summary['input_timesteps']} input scenes; "
            f"sliding-window test={summary['test_timesteps']} scenes; "
            f"validation={summary['val_timesteps']} scenes"
        )
        print(f"Raster size: {summary['height']} x {summary['width']}")
        print(
            f"Temporal split: train={summary['train_timesteps']} scenes, "
            f"test={summary['test_timesteps']} scenes, "
            f"val={summary['val_timesteps']} scenes"
        )
        print(
            f"  train window: input {summary['train_first_input_file']} -> {summary['train_last_input_file']}, "
            f"target {summary['train_first_target_file']} -> {summary['train_last_target_file']}, "
            f"windows={summary['train_windows']}"
        )
        print(
            f"  test target: {summary['test_first_target_file']} -> {summary['test_last_target_file']}"
        )
        print(
            f"  val target: {summary['val_first_target_file']} -> {summary['val_last_target_file']}"
        )
        print(
            f"Spatial mask: full raster for every split, pixels={summary['total_pixels']}"
        )
        print(
            "Normalizer: "
            f"min={normalizer.data_min:.6f}, max={normalizer.data_max:.6f}, "
            f"mean={normalizer.data_mean:.6f}, std={normalizer.data_std:.6f}"
        )
        print(
            f"Parameter layers={summary['parameter_layers']}, "
            f"parameter count={len(summary['parameter_names'])}"
        )
        self.print_memory_estimate(summary)
        self.print_training_parameters()
        print("=" * 80)
        return self.data

    def print_memory_estimate(self, summary: dict) -> None:
        pixels = int(summary["height"]) * int(summary["width"]) * int(self.config.num_channels)
        bytes_per_float = 4
        dense_square_params = pixels * pixels
        reference_dense_params = 2 * dense_square_params
        reference_weight_gib = reference_dense_params * bytes_per_float / 1024**3
        reference_adam_gib = reference_weight_gib * 4

        batch_size = int(self.config.batch_size)
        seq_length = int(self.config.seq_length)
        d_model = int(self.config.informer_d_model)
        n_heads = int(self.config.informer_n_heads)
        d_ff = int(self.config.informer_d_ff)
        temporal_activation_bytes = (
            batch_size * pixels * seq_length * d_model * bytes_per_float * 6
            + batch_size * pixels * n_heads * seq_length * seq_length * bytes_per_float * 2
            + batch_size * pixels * seq_length * d_ff * bytes_per_float
        )
        current_activation_gib = temporal_activation_bytes / 1024**3
        print("VRAM estimate")
        print(
            "  reference dense full-image LGTM: "
            f"pixels={pixels:,}, two N x N layers={reference_dense_params:,} params, "
            f"weights~{reference_weight_gib:.1f} GiB, Adam training state~{reference_adam_gib:.1f} GiB before activations"
        )
        print(
            "  current whole-raster model: no spatial slicing, shared temporal encoder per pixel; "
            f"dominant fp32 activations roughly {current_activation_gib:.1f} GiB/batch before framework overhead"
        )

    def print_training_parameters(self) -> None:
        print("Training parameters")
        print(f"  seq_length={self.config.seq_length}, pred_length={self.config.pred_length}")
        print(
            f"  train_timesteps={self.config.train_timesteps}, "
            f"test_timesteps={self.config.test_timesteps}, "
            f"val_timesteps={self.config.val_timesteps}"
        )
        print(f"  batch_size={self.config.batch_size}, num_epochs={self.config.num_epochs}")
        print(
            f"  learning_rate={self.config.learning_rate}, weight_decay={self.config.weight_decay}, "
            f"lr_scheduler_patience={self.config.lr_scheduler_patience}, "
            f"lr_scheduler_factor={self.config.lr_scheduler_factor}"
        )
        print(f"  patience={self.config.patience}, test_batches={self.config.test_batches}")
        print(
            "  spatial training=whole raster, patch slicing=disabled, "
            f"residual_prediction={getattr(self.config, 'residual_prediction', True)}"
        )
        print(
            f"  loss={self.config.loss_name}, huber_delta={self.config.huber_delta}, "
            f"data_loss_weight={self.config.data_loss_weight}"
        )
        print(
            "  residual constraints: "
            f"residual_loss_weight={self.config.residual_loss_weight}, "
            f"residual_gradient_loss_weight={self.config.residual_gradient_loss_weight}, "
            f"residual_distribution_loss_weight={self.config.residual_distribution_loss_weight}"
        )
        print("  supervised loss: data loss plus normalized adaptive PINN loss")
        print(
            f"  use_pinn_loss={self.config.use_pinn_loss}, "
            f"pinn_loss_weight={self.config.pinn_loss_weight}, "
            f"report_raw_pinn_loss={self.config.report_raw_pinn_loss}, "
            f"load_pinn_parameters={getattr(self.config, 'load_pinn_parameters', False)}, "
            f"pinn_mode={self.config.pinn_mode}"
        )
        if self.config.pinn_mode == "soft3_dominant":
            print(
                "  PINN equation: r3 = w3*dS/dt - lambda3*(Sinf3 - w3*S), "
                "lambda3 = pi^2*cv3/(4*Hdr3^2)"
            )
            print(
                "  PINN units: model displacement and Sinf are in mm; time_deltas are days; "
                f"cv_cm2year_to_m2day={self.config.cv_cm2year_to_m2day}; "
                f"settlement_sign={self.config.settlement_sign}; "
                f"drainage={self.config.soft3_drainage}; "
                f"residual_scale={self.config.pinn_residual_scale_mm_per_day} mm/day"
            )
        print(
            "  PINN adaptive weighting: "
            f"enabled={self.config.use_adaptive_pinn_weighting}, "
            f"initial={self.config.adaptive_pinn_initial_weight}, "
            f"target_ratio={self.config.adaptive_pinn_target_ratio}, "
            f"ema_beta={self.config.adaptive_pinn_ema_beta}, "
            f"bounds=[{self.config.adaptive_pinn_min_weight}, {self.config.adaptive_pinn_max_weight}]"
        )
        print(
            "  gradient-angle PINN control: "
            f"enabled={getattr(self.config, 'use_gradient_angle_pinn_control', False)}, "
            f"warmup_epochs={getattr(self.config, 'gradient_angle_control_warmup_epochs', 0)}, "
            f"cosine_ema_beta={getattr(self.config, 'gradient_angle_cosine_ema_beta', 0.0)}, "
            f"gate_cosine=[{getattr(self.config, 'gradient_gate_negative_cosine', -0.3)}, "
            f"{getattr(self.config, 'gradient_gate_positive_cosine', 0.3)}], "
            f"gate_range=[{getattr(self.config, 'gradient_gate_min_value', 0.05)}, 1.0]"
        )
        print(
            "  gradient conflict diagnostics: "
            f"enabled={self.config.enable_gradient_conflict_diagnostics}, "
            f"split={self.config.gradient_diagnostics_split}, "
            f"interval={self.config.gradient_diagnostics_interval}, "
            f"max_batches={self.config.gradient_diagnostics_max_batches}, "
            f"file={self.config.gradient_diagnostics_filename}"
        )
        print(
            f"  depth_step={self.config.depth_step}, cv_is_log10={self.config.cv_is_log10}, "
            f"cv_cm2s_to_m2day={self.config.cv_cm2s_to_m2day}, "
            f"cv_cm2year_to_m2day={getattr(self.config, 'cv_cm2year_to_m2day', False)}, "
            f"settlement_sign={getattr(self.config, 'settlement_sign', 1.0)}, "
            f"soft3_drainage={getattr(self.config, 'soft3_drainage', 'single')}"
        )
        print(f"  parameter_raster_dir={self.config.parameter_raster_dir}")
        print(
            f"  device={self.config.device}, save_predictions={self.config.save_predictions}, "
            f"future_steps={self.config.future_steps}"
        )
        print(f"  use_pretrained_best_model={USE_PRETRAINED_BEST_MODEL}")

    def build_model(self) -> PatchwiseRasterForecastModel:
        model = PatchwiseRasterForecastModel.from_config(self.config).to(self.device)
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
        self.model = model
        return model

    def train_model(self) -> PatchwiseRasterForecastModel:
        if self.data is None:
            self.setup_data()
        assert self.data is not None

        model = self.model or self.build_model()
        criterion = build_loss(self.config, normalizer=self.data.normalizer).to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        pretrained_state = self.load_pretrained_checkpoint(model, optimizer)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.config.lr_scheduler_patience,
            factor=self.config.lr_scheduler_factor,
        )

        train_loader = self.data.train_loader()
        test_loader = self.data.test_loader()
        val_loader = self.data.val_loader()

        best_loss = pretrained_state["best_loss"]
        no_improve = 0
        train_losses = pretrained_state["train_losses"]
        test_losses = pretrained_state["test_losses"]
        val_losses = pretrained_state["val_losses"]
        best_checkpoint_path: Optional[Path] = None
        pinn_weight_state = self._initial_pinn_weight_state()
        gradient_diagnostics_path = self._gradient_diagnostics_path()
        gradient_diagnostics_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_metrics_path = self.config.output_dir / "training_epoch_metrics.csv"
        if self.config.enable_gradient_conflict_diagnostics:
            print(
                "Gradient conflict diagnostics enabled: "
                f"split={self.config.gradient_diagnostics_split}, "
                f"interval={self.config.gradient_diagnostics_interval}, "
                f"max_batches={self.config.gradient_diagnostics_max_batches}, "
                f"output={gradient_diagnostics_path}"
            )

        try:
            for epoch in range(self.config.num_epochs):
                if self.config.use_adaptive_pinn_weighting:
                    active_pinn_weight = float(pinn_weight_state["weight"])
                else:
                    active_pinn_weight = self._pinn_loss_weight_for_epoch(epoch)
                    pinn_weight_state["weight"] = active_pinn_weight
                    pinn_weight_state["raw_weight"] = active_pinn_weight
                criterion.pinn_loss_weight = active_pinn_weight

                train_metrics = self._train_one_epoch(model, criterion, optimizer, train_loader, epoch)
                train_loss = train_metrics["total_loss"]
                test_1step_metrics = self.evaluate_model(model, criterion, test_loader, rollout=False)
                test_loss = test_1step_metrics["total_loss"]
                val_1step_metrics = self.evaluate_model(model, criterion, val_loader, rollout=False)
                val_loss = val_1step_metrics["total_loss"]
                scheduler.step(val_loss)
                gradient_rows = self._collect_gradient_conflict_rows(
                    model=model,
                    criterion=criterion,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    val_loader=val_loader,
                    epoch=epoch,
                    active_pinn_weight=active_pinn_weight,
                    next_pinn_weight=active_pinn_weight,
                    pinn_weight_state=pinn_weight_state,
                    train_metrics=train_metrics,
                    run_id=gradient_diagnostics_run_id,
                )
                gradient_summary_for_control = self._summarize_gradient_rows(gradient_rows)
                pinn_weight_state = self._update_adaptive_pinn_weight(
                    train_metrics,
                    pinn_weight_state,
                    epoch=epoch,
                    gradient_summary=gradient_summary_for_control,
                )
                self._refresh_gradient_next_weight_fields(
                    gradient_rows,
                    next_pinn_weight=float(pinn_weight_state["weight"]),
                    pinn_weight_state=pinn_weight_state,
                )
                if gradient_rows:
                    self._append_gradient_diagnostics_rows(gradient_diagnostics_path, gradient_rows)
                gradient_summary = self._summarize_gradient_rows(gradient_rows)

                train_losses.append(float(train_loss))
                test_losses.append(float(test_loss))
                val_losses.append(float(val_loss))
                self._trim_loss_history(train_losses)
                self._trim_loss_history(test_losses)
                self._trim_loss_history(val_losses)
                epoch_row = {
                    "run_id": gradient_diagnostics_run_id,
                    "epoch": epoch + 1,
                    "train_total_loss": train_metrics["total_loss"],
                    "train_data_loss": train_metrics["data_loss"],
                    "train_raw_data_loss": train_metrics["raw_data_loss"],
                    "train_pinn_loss": train_metrics["pinn_loss"],
                    "train_raw_pinn_loss": train_metrics["raw_pinn_loss"],
                    "test_total_loss": test_1step_metrics["total_loss"],
                    "test_data_loss": test_1step_metrics["data_loss"],
                    "test_raw_data_loss": test_1step_metrics["raw_data_loss"],
                    "test_pinn_loss": test_1step_metrics["pinn_loss"],
                    "test_raw_pinn_loss": test_1step_metrics["raw_pinn_loss"],
                    "val_total_loss": val_1step_metrics["total_loss"],
                    "val_data_loss": val_1step_metrics["data_loss"],
                    "val_raw_data_loss": val_1step_metrics["raw_data_loss"],
                    "val_pinn_loss": val_1step_metrics["pinn_loss"],
                    "val_raw_pinn_loss": val_1step_metrics["raw_pinn_loss"],
                    "pinn_weight": active_pinn_weight,
                    "next_pinn_weight": pinn_weight_state["weight"],
                    "adaptive_raw_weight": pinn_weight_state["raw_weight"],
                    "adaptive_base_weight": pinn_weight_state["base_weight"],
                    "adaptive_base_raw_weight": pinn_weight_state["base_raw_weight"],
                    "gradient_gate": pinn_weight_state["gradient_gate"],
                    "gradient_cosine_ema": pinn_weight_state["gradient_cosine_ema"],
                    "gradient_control_active": pinn_weight_state["gradient_control_active"],
                    "adaptive_data_reference": pinn_weight_state["data_reference"],
                    "adaptive_pinn_reference": pinn_weight_state["pinn_reference"],
                    "adaptive_norm_data": pinn_weight_state["normalized_data_loss"],
                    "adaptive_norm_pinn": pinn_weight_state["normalized_pinn_loss"],
                    "grad_cosine_data_pinn": gradient_summary["grad_cosine_data_pinn"],
                    "gradient_angle_degrees": gradient_summary["gradient_angle_degrees"],
                    "gradient_conflict_fraction": gradient_summary["gradient_conflict_fraction"],
                    "grad_norm_ratio_pinn_data": gradient_summary["grad_norm_ratio_pinn_data"],
                    "next_weighted_raw_grad_norm_ratio_pinn_data": gradient_summary[
                        "next_weighted_raw_grad_norm_ratio_pinn_data"
                    ],
                    "gradient_diagnostic_batches": gradient_summary["gradient_diagnostic_batches"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
                self._append_training_epoch_row(epoch_metrics_path, epoch_row)
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                    f"train_total_loss={train_metrics['total_loss']:.6f}, "
                    f"train_data_loss={train_metrics['data_loss']:.6f}, "
                    f"train_raw_data_loss={train_metrics['raw_data_loss']:.6f}, "
                    f"train_PINN_loss={train_metrics['pinn_loss']:.6f}, "
                    f"train_raw_PINN_loss={train_metrics['raw_pinn_loss']:.6f}, "
                    f"test_1step_total_loss={test_1step_metrics['total_loss']:.6f}, "
                    f"test_1step_data_loss={test_1step_metrics['data_loss']:.6f}, "
                    f"test_1step_raw_PINN_loss={test_1step_metrics['raw_pinn_loss']:.6f}, "
                    f"val_1step_total_loss={val_1step_metrics['total_loss']:.6f}, "
                    f"val_1step_data_loss={val_1step_metrics['data_loss']:.6f}, "
                    f"val_1step_raw_PINN_loss={val_1step_metrics['raw_pinn_loss']:.6f}, "
                    f"pinn_weight={active_pinn_weight:.2e}, "
                    f"base_next_pinn_weight={pinn_weight_state['base_weight']:.2e}, "
                    f"gradient_gate={pinn_weight_state['gradient_gate']:.3f}, "
                    f"gradient_cosine_ema={pinn_weight_state['gradient_cosine_ema']:.6f}, "
                    f"next_pinn_weight={pinn_weight_state['weight']:.2e}, "
                    f"adaptive_norm_data={pinn_weight_state['normalized_data_loss']:.6f}, "
                    f"adaptive_norm_PINN={pinn_weight_state['normalized_pinn_loss']:.6f}, "
                    f"grad_cosine_data_PINN={gradient_summary['grad_cosine_data_pinn']:.6f}, "
                    f"gradient_angle_degrees={gradient_summary['gradient_angle_degrees']:.6f}, "
                    f"gradient_conflict_fraction={gradient_summary['gradient_conflict_fraction']:.6f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve = 0
                    best_checkpoint_path = self.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=best_loss,
                        train_losses=train_losses,
                        test_losses=test_losses,
                        val_losses=val_losses,
                        filename="best_lgtm_lager_pinn_gradgate_model.pth",
                    )
                    print(f"Saved best model: {best_loss:.6f}")
                else:
                    no_improve += 1

                if self.config.checkpoint_interval and (epoch + 1) % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=test_loss,
                        train_losses=train_losses,
                        test_losses=test_losses,
                        val_losses=val_losses,
                        filename=f"checkpoint_epoch_{epoch + 1}.pth",
                    )

                self._epoch_memory_cleanup(epoch)

                if no_improve >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}. Best val_loss={best_loss:.6f}")
                    break
        finally:
            del train_loader, test_loader, val_loader, criterion, scheduler, optimizer
            self.cleanup_memory()

        self.plot_loss_curves(train_losses, test_losses, val_losses)
        if best_checkpoint_path is not None:
            model = self.load_model(best_checkpoint_path)
            print(f"Loaded best model for downstream prediction: {best_checkpoint_path}")
        else:
            self.model = model
        return model

    def plot_loss_curves(self, train_losses, test_losses, val_losses=None) -> Optional[Path]:
        val_losses = val_losses or []
        if not train_losses and not test_losses and not val_losses:
            print("No loss history recorded; skipped loss plot.")
            return None

        output_path = self.config.output_dir / "training_loss.png"
        epochs = np.arange(1, max(len(train_losses), len(test_losses), len(val_losses)) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        if train_losses:
            plt.plot(epochs[: len(train_losses)], train_losses, label="Training loss", linewidth=2)
        if test_losses:
            plt.plot(epochs[: len(test_losses)], test_losses, label="Test loss", linewidth=2)
        if val_losses:
            plt.plot(epochs[: len(val_losses)], val_losses, label="Validation loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        if train_losses:
            plt.semilogy(epochs[: len(train_losses)], train_losses, label="Training loss", linewidth=2)
        if test_losses:
            plt.semilogy(epochs[: len(test_losses)], test_losses, label="Test loss", linewidth=2)
        if val_losses:
            plt.semilogy(epochs[: len(val_losses)], val_losses, label="Validation loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss (Log Scale)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved loss plot: {output_path}")
        return output_path

    def _trim_loss_history(self, history: list) -> None:
        limit = self.config.loss_history_limit
        if len(history) > limit:
            del history[:-limit]

    def _epoch_memory_cleanup(self, epoch: int) -> None:
        epoch_number = epoch + 1
        if self.config.clear_reader_cache_each_epoch and self.data is not None:
            self.data.clear_runtime_caches()
        if self.config.epoch_cleanup_interval and epoch_number % self.config.epoch_cleanup_interval == 0:
            self.cleanup_memory()

    def cleanup_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def load_pretrained_checkpoint(self, model, optimizer) -> dict:
        state = {
            "best_loss": float("inf"),
            "train_losses": [],
            "test_losses": [],
            "val_losses": [],
        }
        if not USE_PRETRAINED_BEST_MODEL:
            print("Pretrained loading disabled; training starts from random initialization.")
            return state

        checkpoint_path = self.config.model_dir / "best_lgtm_lager_pinn_gradgate_model.pth"
        if not checkpoint_path.exists():
            print(f"Pretrained checkpoint not found: {checkpoint_path}. Training starts from random initialization.")
            return state

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=STRICT_PRETRAINED_LOAD)
        print(f"Loaded pretrained model weights from: {checkpoint_path}")

        if RESUME_OPTIMIZER_STATE and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state from pretrained checkpoint.")
        else:
            print("Optimizer state was not restored; optimizer starts fresh.")

        if KEEP_BEST_LOSS_FROM_CHECKPOINT:
            state["best_loss"] = float(checkpoint.get("loss", float("inf")))
            state["train_losses"] = list(checkpoint.get("train_losses", []))
            state["test_losses"] = list(checkpoint.get("test_losses", checkpoint.get("val_losses", [])))
            state["val_losses"] = list(checkpoint.get("val_losses", []))
            print(f"Kept best loss from checkpoint: {state['best_loss']:.6f}")
        del checkpoint
        self.cleanup_memory()
        return state

    def _pinn_loss_weight_for_epoch(self, epoch: int) -> float:
        if not self.config.use_pinn_loss:
            return 0.0
        if self.config.use_adaptive_pinn_weighting:
            return float(self.config.adaptive_pinn_initial_weight)
        return float(self.config.pinn_loss_weight)

    def _initial_pinn_weight_state(self) -> dict:
        initial_weight = self._pinn_loss_weight_for_epoch(0)
        return {
            "weight": initial_weight,
            "raw_weight": initial_weight,
            "base_weight": initial_weight,
            "base_raw_weight": initial_weight,
            "gradient_gate": 1.0,
            "gradient_cosine_ema": float("nan"),
            "gradient_control_active": 0.0,
            "data_reference": float("nan"),
            "pinn_reference": float("nan"),
            "normalized_data_loss": float("nan"),
            "normalized_pinn_loss": float("nan"),
        }

    def _update_adaptive_pinn_weight(
        self,
        metrics: dict,
        previous_state: dict,
        epoch: int,
        gradient_summary: Optional[dict] = None,
    ) -> dict:
        if not self.config.use_pinn_loss:
            state = dict(previous_state)
            state["weight"] = 0.0
            state["raw_weight"] = 0.0
            state["base_weight"] = 0.0
            state["base_raw_weight"] = 0.0
            state["gradient_gate"] = 0.0
            state["gradient_control_active"] = 0.0
            return state
        if not self.config.use_adaptive_pinn_weighting:
            return dict(previous_state)

        epsilon = float(self.config.adaptive_pinn_epsilon)
        raw_data_loss = float(metrics.get("raw_data_loss", metrics.get("data_loss", 0.0)))
        raw_pinn_loss = float(metrics.get("raw_pinn_loss", 0.0))
        data_reference = self._positive_reference(
            previous_state.get("data_reference", float("nan")),
            raw_data_loss,
            epsilon,
        )
        pinn_reference = self._positive_reference(
            previous_state.get("pinn_reference", float("nan")),
            raw_pinn_loss,
            epsilon,
        )
        normalized_data = raw_data_loss / max(data_reference, epsilon)
        normalized_pinn = raw_pinn_loss / max(pinn_reference, epsilon)
        raw_weight = (
            float(self.config.data_loss_weight)
            * float(self.config.adaptive_pinn_target_ratio)
            * normalized_data
            / max(normalized_pinn, epsilon)
            * data_reference
            / max(pinn_reference, epsilon)
        )
        base_raw_weight = self._finite_or_default(
            raw_weight,
            previous_state.get("base_raw_weight", previous_state.get("raw_weight", self.config.adaptive_pinn_initial_weight)),
        )
        base_raw_weight = float(
            np.clip(
                base_raw_weight,
                self.config.adaptive_pinn_min_weight,
                self.config.adaptive_pinn_max_weight,
            )
        )
        previous_base_weight = float(previous_state.get("base_weight", previous_state.get("weight", base_raw_weight)))
        beta = float(self.config.adaptive_pinn_ema_beta)
        base_weight = beta * previous_base_weight + (1.0 - beta) * base_raw_weight
        base_weight = self._finite_or_default(base_weight, previous_base_weight)
        base_weight = float(
            np.clip(
                base_weight,
                self.config.adaptive_pinn_min_weight,
                self.config.adaptive_pinn_max_weight,
            )
        )
        previous_weight = float(previous_state.get("weight", base_weight))
        gate_state = self._gradient_angle_gate_state(
            epoch=epoch,
            gradient_summary=gradient_summary,
            previous_state=previous_state,
        )
        gated_target_weight = base_weight * gate_state["gradient_gate"]
        if gate_state["gradient_control_active"] > 0.0:
            weight = beta * previous_weight + (1.0 - beta) * gated_target_weight
        else:
            weight = base_weight
        weight = self._finite_or_default(weight, previous_weight)
        weight = float(
            np.clip(
                weight,
                self.config.adaptive_pinn_min_weight,
                self.config.adaptive_pinn_max_weight,
            )
        )
        return {
            "weight": weight,
            "raw_weight": base_raw_weight,
            "base_weight": base_weight,
            "base_raw_weight": base_raw_weight,
            "gradient_gate": gate_state["gradient_gate"],
            "gradient_cosine_ema": gate_state["gradient_cosine_ema"],
            "gradient_control_active": gate_state["gradient_control_active"],
            "data_reference": data_reference,
            "pinn_reference": pinn_reference,
            "normalized_data_loss": normalized_data,
            "normalized_pinn_loss": normalized_pinn,
        }

    def _gradient_angle_gate_state(
        self,
        epoch: int,
        gradient_summary: Optional[dict],
        previous_state: dict,
    ) -> dict:
        previous_cosine_ema = float(previous_state.get("gradient_cosine_ema", float("nan")))
        state = {
            "gradient_gate": 1.0,
            "gradient_cosine_ema": previous_cosine_ema,
            "gradient_control_active": 0.0,
        }
        if not getattr(self.config, "use_gradient_angle_pinn_control", False):
            return state
        if not self.config.enable_gradient_conflict_diagnostics:
            return state
        if gradient_summary is None:
            return state

        cosine = float(gradient_summary.get("grad_cosine_data_pinn", float("nan")))
        if not np.isfinite(cosine):
            return state

        cosine = float(np.clip(cosine, -1.0, 1.0))
        beta = float(getattr(self.config, "gradient_angle_cosine_ema_beta", 0.8))
        if np.isfinite(previous_cosine_ema):
            cosine_ema = beta * previous_cosine_ema + (1.0 - beta) * cosine
        else:
            cosine_ema = cosine
        cosine_ema = float(np.clip(cosine_ema, -1.0, 1.0))
        state["gradient_cosine_ema"] = cosine_ema

        warmup_epochs = int(getattr(self.config, "gradient_angle_control_warmup_epochs", 0))
        if epoch + 1 <= warmup_epochs:
            return state

        state["gradient_gate"] = self._gradient_gate_from_cosine(cosine_ema)
        state["gradient_control_active"] = 1.0
        return state

    def _gradient_gate_from_cosine(self, cosine: float) -> float:
        if not np.isfinite(cosine):
            return 1.0
        positive = float(getattr(self.config, "gradient_gate_positive_cosine", 0.3))
        negative = float(getattr(self.config, "gradient_gate_negative_cosine", -0.3))
        neutral_gate = float(getattr(self.config, "gradient_gate_neutral_value", 0.5))
        min_gate = float(getattr(self.config, "gradient_gate_min_value", 0.05))

        cosine = float(np.clip(cosine, -1.0, 1.0))
        if cosine >= positive:
            return 1.0
        if cosine >= 0.0:
            ratio = cosine / max(positive, self.config.adaptive_pinn_epsilon)
            return float(neutral_gate + ratio * (1.0 - neutral_gate))
        if cosine <= negative:
            return min_gate
        ratio = (cosine - negative) / max(0.0 - negative, self.config.adaptive_pinn_epsilon)
        return float(min_gate + ratio * (neutral_gate - min_gate))

    def _positive_reference(self, previous: float, current: float, epsilon: float) -> float:
        current = float(current)
        previous = float(previous)
        if not np.isfinite(current) or current <= epsilon:
            return previous if np.isfinite(previous) and previous > epsilon else epsilon
        if not np.isfinite(previous) or previous <= epsilon:
            return current
        beta = float(self.config.adaptive_pinn_ema_beta)
        return beta * previous + (1.0 - beta) * current

    def _finite_or_default(self, value: float, default: float) -> float:
        value = float(value)
        return value if np.isfinite(value) else float(default)

    def _gradient_diagnostics_path(self) -> Path:
        return self.config.output_dir / self.config.gradient_diagnostics_filename

    def _collect_gradient_conflict_rows(
        self,
        model,
        criterion,
        train_loader,
        test_loader,
        val_loader,
        epoch: int,
        active_pinn_weight: float,
        next_pinn_weight: float,
        pinn_weight_state: dict,
        train_metrics: dict,
        run_id: str,
    ) -> list:
        if not self.config.enable_gradient_conflict_diagnostics:
            return []
        if (epoch + 1) % self.config.gradient_diagnostics_interval != 0:
            return []

        split_loaders = {
            "train": train_loader,
            "test": test_loader,
            "val": val_loader,
        }
        requested = self.config.gradient_diagnostics_split
        split_names = ("train", "test", "val") if requested == "all" else (requested,)
        rows = []
        for split_name in split_names:
            rows.extend(
                self._gradient_conflict_rows_for_split(
                    model=model,
                    criterion=criterion,
                    loader=split_loaders[split_name],
                    split_name=split_name,
                    epoch=epoch,
                    active_pinn_weight=active_pinn_weight,
                    next_pinn_weight=next_pinn_weight,
                    pinn_weight_state=pinn_weight_state,
                    train_metrics=train_metrics,
                    run_id=run_id,
                )
            )
        return rows

    def _run_gradient_conflict_diagnostics(
        self,
        model,
        criterion,
        train_loader,
        test_loader,
        val_loader,
        epoch: int,
        active_pinn_weight: float,
        next_pinn_weight: float,
        pinn_weight_state: dict,
        train_metrics: dict,
        output_path: Path,
        run_id: str,
    ) -> dict:
        rows = self._collect_gradient_conflict_rows(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            epoch=epoch,
            active_pinn_weight=active_pinn_weight,
            next_pinn_weight=next_pinn_weight,
            pinn_weight_state=pinn_weight_state,
            train_metrics=train_metrics,
            run_id=run_id,
        )
        if rows:
            self._append_gradient_diagnostics_rows(output_path, rows)
        return self._summarize_gradient_rows(rows)

    def _gradient_conflict_rows_for_split(
        self,
        model,
        criterion,
        loader,
        split_name: str,
        epoch: int,
        active_pinn_weight: float,
        next_pinn_weight: float,
        pinn_weight_state: dict,
        train_metrics: dict,
        run_id: str,
    ) -> list:
        rows = []
        original_pinn_weight = criterion.pinn_loss_weight
        was_training = model.training
        diagnostic_loader = self._gradient_diagnostic_loader(loader, split_name)
        model.eval()
        model.zero_grad(set_to_none=True)
        try:
            max_batches = int(self.config.gradient_diagnostics_max_batches)
            for batch_index, batch in enumerate(diagnostic_loader):
                if batch_index >= max_batches:
                    break
                rows.append(
                    self._gradient_conflict_metrics_for_batch(
                        model=model,
                        criterion=criterion,
                        batch=batch,
                        split_name=split_name,
                        epoch=epoch,
                        batch_index=batch_index,
                        active_pinn_weight=active_pinn_weight,
                        next_pinn_weight=next_pinn_weight,
                        pinn_weight_state=pinn_weight_state,
                        train_metrics=train_metrics,
                        run_id=run_id,
                    )
                )
                model.zero_grad(set_to_none=True)
        finally:
            criterion.pinn_loss_weight = original_pinn_weight
            model.train(was_training)
            model.zero_grad(set_to_none=True)
            self.cleanup_memory()
        return rows

    def _gradient_diagnostic_loader(self, fallback_loader, split_name: str):
        if self.data is None:
            return fallback_loader
        dataset = getattr(self.data, f"{split_name}_dataset", None)
        if dataset is None:
            return fallback_loader
        try:
            return self.data._build_loader(dataset, batch_size=self.config.batch_size, shuffle=False)
        except Exception as exc:
            print(f"Warning: failed to build deterministic {split_name} diagnostic loader: {exc}", file=sys.stderr)
            return fallback_loader

    def _gradient_conflict_metrics_for_batch(
        self,
        model,
        criterion,
        batch,
        split_name: str,
        epoch: int,
        batch_index: int,
        active_pinn_weight: float,
        next_pinn_weight: float,
        pinn_weight_state: dict,
        train_metrics: dict,
        run_id: str,
    ) -> dict:
        inputs, targets, _target_indexes, parameters, mask, time_deltas = self._prepare_batch(batch)
        predictions = model(inputs)
        loss_parts = criterion.components(
            prediction=predictions,
            target=targets[:, : predictions.shape[1]],
            mask=mask,
            parameters=parameters,
            input_sequence=inputs,
            time_deltas=time_deltas[:, : predictions.shape[1]] if time_deltas.dim() == 2 else time_deltas[: predictions.shape[1]],
        )
        self._assert_finite_loss_parts(loss_parts, f"gradient_diagnostics epoch={epoch + 1}, batch={batch_index + 1}")

        epsilon = float(self.config.adaptive_pinn_epsilon)
        raw_data_loss_value = float(loss_parts["raw_data_loss"].detach().cpu().item())
        raw_pinn_loss_value = float(loss_parts["raw_pinn_loss"].detach().cpu().item())
        data_reference = self._diagnostic_reference(
            pinn_weight_state.get("data_reference", float("nan")),
            train_metrics.get("raw_data_loss", float("nan")),
            raw_data_loss_value,
            epsilon,
        )
        pinn_reference = self._diagnostic_reference(
            pinn_weight_state.get("pinn_reference", float("nan")),
            train_metrics.get("raw_pinn_loss", float("nan")),
            raw_pinn_loss_value,
            epsilon,
        )

        normalized_data_loss = loss_parts["raw_data_loss"] / max(data_reference, epsilon)
        normalized_pinn_loss = loss_parts["raw_pinn_loss"] / max(pinn_reference, epsilon)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        data_grads = self._autograd_grad(normalized_data_loss, params, retain_graph=True)
        pinn_grads = self._autograd_grad(normalized_pinn_loss, params, retain_graph=False)
        grad_stats = self._gradient_pair_stats(data_grads, pinn_grads)

        normalized_grad_ratio = grad_stats["grad_norm_ratio_pinn_data"]
        raw_grad_ratio = (
            grad_stats["grad_norm_pinn"] * pinn_reference / max(grad_stats["grad_norm_data"] * data_reference, epsilon)
            if np.isfinite(grad_stats["grad_norm_pinn"]) and np.isfinite(grad_stats["grad_norm_data"])
            else float("nan")
        )
        data_loss_weight = max(float(getattr(self.config, "data_loss_weight", 1.0)), epsilon)
        active_weighted_normalized_ratio = float(active_pinn_weight) * normalized_grad_ratio / data_loss_weight
        next_weighted_normalized_ratio = float(next_pinn_weight) * normalized_grad_ratio / data_loss_weight
        active_weighted_raw_ratio = float(active_pinn_weight) * raw_grad_ratio / data_loss_weight
        next_weighted_raw_ratio = float(next_pinn_weight) * raw_grad_ratio / data_loss_weight
        cosine = grad_stats["grad_cosine_data_pinn"]
        angle_degrees = (
            float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            if np.isfinite(cosine)
            else float("nan")
        )

        row = {
            "run_id": run_id,
            "epoch": epoch + 1,
            "split": split_name,
            "batch_index": batch_index + 1,
            "active_pinn_weight": float(active_pinn_weight),
            "next_pinn_weight": float(next_pinn_weight),
            "adaptive_raw_weight": float(pinn_weight_state.get("raw_weight", float("nan"))),
            "adaptive_base_weight": float(pinn_weight_state.get("base_weight", float("nan"))),
            "adaptive_base_raw_weight": float(pinn_weight_state.get("base_raw_weight", float("nan"))),
            "gradient_gate": float(pinn_weight_state.get("gradient_gate", float("nan"))),
            "gradient_cosine_ema": float(pinn_weight_state.get("gradient_cosine_ema", float("nan"))),
            "gradient_control_active": float(pinn_weight_state.get("gradient_control_active", float("nan"))),
            "data_reference": data_reference,
            "pinn_reference": pinn_reference,
            "raw_data_loss": raw_data_loss_value,
            "raw_pinn_loss": raw_pinn_loss_value,
            "normalized_data_loss": raw_data_loss_value / max(data_reference, epsilon),
            "normalized_pinn_loss": raw_pinn_loss_value / max(pinn_reference, epsilon),
            "grad_norm_data": grad_stats["grad_norm_data"],
            "grad_norm_pinn": grad_stats["grad_norm_pinn"],
            "grad_dot_data_pinn": grad_stats["grad_dot_data_pinn"],
            "grad_cosine_data_pinn": cosine,
            "gradient_angle_degrees": angle_degrees,
            "is_gradient_conflict": 1 if np.isfinite(cosine) and cosine < 0.0 else 0,
            "grad_norm_ratio_pinn_data": normalized_grad_ratio,
            "raw_grad_norm_ratio_pinn_data": raw_grad_ratio,
            "active_weighted_normalized_grad_norm_ratio_pinn_data": active_weighted_normalized_ratio,
            "next_weighted_normalized_grad_norm_ratio_pinn_data": next_weighted_normalized_ratio,
            "active_weighted_raw_grad_norm_ratio_pinn_data": active_weighted_raw_ratio,
            "next_weighted_raw_grad_norm_ratio_pinn_data": next_weighted_raw_ratio,
        }
        del inputs, targets, parameters, mask, time_deltas, predictions, loss_parts
        del normalized_data_loss, normalized_pinn_loss, data_grads, pinn_grads
        return row

    def _diagnostic_reference(self, *values: float) -> float:
        epsilon = float(values[-1])
        for value in values[:-1]:
            value = float(value)
            if np.isfinite(value) and value > epsilon:
                return value
        return epsilon

    def _autograd_grad(self, loss: torch.Tensor, params: list, retain_graph: bool) -> list:
        if not torch.is_tensor(loss) or not loss.requires_grad:
            return [None for _ in params]
        return list(
            torch.autograd.grad(
                loss,
                params,
                retain_graph=retain_graph,
                allow_unused=True,
            )
        )

    def _gradient_pair_stats(self, data_grads: list, pinn_grads: list) -> dict:
        dot_value = 0.0
        data_sq = 0.0
        pinn_sq = 0.0
        for data_grad, pinn_grad in zip(data_grads, pinn_grads):
            if data_grad is not None:
                data_values = data_grad.detach()
                data_sq += float(torch.sum(data_values * data_values).cpu().item())
            else:
                data_values = None
            if pinn_grad is not None:
                pinn_values = pinn_grad.detach()
                pinn_sq += float(torch.sum(pinn_values * pinn_values).cpu().item())
            else:
                pinn_values = None
            if data_values is not None and pinn_values is not None:
                dot_value += float(torch.sum(data_values * pinn_values).cpu().item())

        epsilon = float(self.config.adaptive_pinn_epsilon)
        data_norm = float(np.sqrt(max(data_sq, 0.0)))
        pinn_norm = float(np.sqrt(max(pinn_sq, 0.0)))
        if data_norm > epsilon and pinn_norm > epsilon:
            cosine = dot_value / (data_norm * pinn_norm)
            cosine = float(np.clip(cosine, -1.0, 1.0))
        else:
            cosine = float("nan")
        return {
            "grad_norm_data": data_norm,
            "grad_norm_pinn": pinn_norm,
            "grad_dot_data_pinn": dot_value,
            "grad_cosine_data_pinn": cosine,
            "grad_norm_ratio_pinn_data": pinn_norm / max(data_norm, epsilon),
        }

    def _append_gradient_diagnostics_rows(self, path: Path, rows: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id",
            "epoch",
            "split",
            "batch_index",
            "active_pinn_weight",
            "next_pinn_weight",
            "adaptive_raw_weight",
            "adaptive_base_weight",
            "adaptive_base_raw_weight",
            "gradient_gate",
            "gradient_cosine_ema",
            "gradient_control_active",
            "data_reference",
            "pinn_reference",
            "raw_data_loss",
            "raw_pinn_loss",
            "normalized_data_loss",
            "normalized_pinn_loss",
            "grad_norm_data",
            "grad_norm_pinn",
            "grad_dot_data_pinn",
            "grad_cosine_data_pinn",
            "gradient_angle_degrees",
            "is_gradient_conflict",
            "grad_norm_ratio_pinn_data",
            "raw_grad_norm_ratio_pinn_data",
            "active_weighted_normalized_grad_norm_ratio_pinn_data",
            "next_weighted_normalized_grad_norm_ratio_pinn_data",
            "active_weighted_raw_grad_norm_ratio_pinn_data",
            "next_weighted_raw_grad_norm_ratio_pinn_data",
        ]
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _summarize_gradient_rows(self, rows: list) -> dict:
        summary = self._empty_gradient_summary()
        numeric_keys = [
            "grad_cosine_data_pinn",
            "grad_norm_ratio_pinn_data",
            "raw_grad_norm_ratio_pinn_data",
            "active_weighted_normalized_grad_norm_ratio_pinn_data",
            "next_weighted_normalized_grad_norm_ratio_pinn_data",
            "active_weighted_raw_grad_norm_ratio_pinn_data",
            "next_weighted_raw_grad_norm_ratio_pinn_data",
            "gradient_angle_degrees",
        ]
        for key in numeric_keys:
            values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
            if values:
                summary[key] = float(np.mean(values))
        conflict_values = [float(row["is_gradient_conflict"]) for row in rows]
        if conflict_values:
            summary["gradient_conflict_fraction"] = float(np.mean(conflict_values))
        summary["gradient_diagnostic_batches"] = float(len(rows))
        summary["weighted_grad_norm_ratio_pinn_data"] = summary[
            "next_weighted_normalized_grad_norm_ratio_pinn_data"
        ]
        return summary

    def _empty_gradient_summary(self) -> dict:
        return {
            "grad_cosine_data_pinn": float("nan"),
            "grad_norm_ratio_pinn_data": float("nan"),
            "raw_grad_norm_ratio_pinn_data": float("nan"),
            "active_weighted_normalized_grad_norm_ratio_pinn_data": float("nan"),
            "next_weighted_normalized_grad_norm_ratio_pinn_data": float("nan"),
            "active_weighted_raw_grad_norm_ratio_pinn_data": float("nan"),
            "next_weighted_raw_grad_norm_ratio_pinn_data": float("nan"),
            "weighted_grad_norm_ratio_pinn_data": float("nan"),
            "gradient_angle_degrees": float("nan"),
            "gradient_conflict_fraction": float("nan"),
            "gradient_diagnostic_batches": 0.0,
        }

    def _refresh_gradient_next_weight_fields(
        self,
        rows: list,
        next_pinn_weight: float,
        pinn_weight_state: dict,
    ) -> None:
        if not rows:
            return
        data_loss_weight = max(float(getattr(self.config, "data_loss_weight", 1.0)), self.config.adaptive_pinn_epsilon)
        for row in rows:
            row["next_pinn_weight"] = float(next_pinn_weight)
            row["adaptive_raw_weight"] = float(pinn_weight_state.get("raw_weight", float("nan")))
            row["adaptive_base_weight"] = float(pinn_weight_state.get("base_weight", float("nan")))
            row["adaptive_base_raw_weight"] = float(pinn_weight_state.get("base_raw_weight", float("nan")))
            row["gradient_gate"] = float(pinn_weight_state.get("gradient_gate", float("nan")))
            row["gradient_cosine_ema"] = float(pinn_weight_state.get("gradient_cosine_ema", float("nan")))
            row["gradient_control_active"] = float(pinn_weight_state.get("gradient_control_active", float("nan")))

            normalized_ratio = float(row.get("grad_norm_ratio_pinn_data", float("nan")))
            raw_ratio = float(row.get("raw_grad_norm_ratio_pinn_data", float("nan")))
            row["next_weighted_normalized_grad_norm_ratio_pinn_data"] = (
                float(next_pinn_weight) * normalized_ratio / data_loss_weight
                if np.isfinite(normalized_ratio)
                else float("nan")
            )
            row["next_weighted_raw_grad_norm_ratio_pinn_data"] = (
                float(next_pinn_weight) * raw_ratio / data_loss_weight
                if np.isfinite(raw_ratio)
                else float("nan")
            )

    def _append_training_epoch_row(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _train_one_epoch(self, model, criterion, optimizer, train_loader, epoch: int) -> dict:
        model.train()
        running_total_loss = 0.0
        running_data_loss = 0.0
        running_raw_data_loss = 0.0
        running_residual_loss = 0.0
        running_residual_gradient_loss = 0.0
        running_residual_distribution_loss = 0.0
        running_pinn_loss = 0.0
        running_raw_pinn_loss = 0.0
        batch_count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
        for batch in progress:
            inputs, targets, _target_indexes, parameters, mask, time_deltas = self._prepare_batch(batch)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss_parts = criterion.components(
                prediction=predictions,
                target=targets,
                mask=mask,
                parameters=parameters,
                input_sequence=inputs,
                time_deltas=time_deltas,
            )
            self._assert_finite_loss_parts(loss_parts, f"epoch={epoch + 1}, batch={batch_count + 1}")
            loss = loss_parts["total_loss"]
            loss.backward()
            self._assert_finite_gradients(model, f"epoch={epoch + 1}, batch={batch_count + 1}")
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.config.gradient_clip,
                    error_if_nonfinite=True,
                )
            optimizer.step()

            running_total_loss += float(loss_parts["total_loss"].item())
            running_data_loss += float(loss_parts["data_loss"].item())
            running_raw_data_loss += float(loss_parts["raw_data_loss"].item())
            running_residual_loss += float(loss_parts["residual_loss"].item())
            running_residual_gradient_loss += float(loss_parts["residual_gradient_loss"].item())
            running_residual_distribution_loss += float(loss_parts["residual_distribution_loss"].item())
            running_pinn_loss += float(loss_parts["pinn_loss"].item())
            running_raw_pinn_loss += float(loss_parts["raw_pinn_loss"].item())
            batch_count += 1
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    total=f"{loss_parts['total_loss'].item():.6f}",
                    data=f"{loss_parts['data_loss'].item():.6f}",
                    residual=f"{loss_parts['residual_loss'].item():.6f}",
                    grad=f"{loss_parts['residual_gradient_loss'].item():.6f}",
                    PINN=f"{loss_parts['pinn_loss'].item():.6f}",
                    raw_PINN=f"{loss_parts['raw_pinn_loss'].item():.6f}",
                )
            del inputs, targets, parameters, predictions, loss, loss_parts, mask, time_deltas
            if self.config.batch_cleanup_interval and batch_count % self.config.batch_cleanup_interval == 0:
                self.cleanup_memory()

        if batch_count == 0:
            raise RuntimeError("Train loader produced no batches.")
        self.cleanup_memory()
        return {
            "total_loss": running_total_loss / batch_count,
            "data_loss": running_data_loss / batch_count,
            "raw_data_loss": running_raw_data_loss / batch_count,
            "residual_loss": running_residual_loss / batch_count,
            "residual_gradient_loss": running_residual_gradient_loss / batch_count,
            "residual_distribution_loss": running_residual_distribution_loss / batch_count,
            "pinn_loss": running_pinn_loss / batch_count,
            "raw_pinn_loss": running_raw_pinn_loss / batch_count,
        }

    def _assert_finite_loss_parts(self, loss_parts: dict, context: str) -> None:
        bad_parts = []
        for name, value in loss_parts.items():
            if torch.is_tensor(value) and not torch.isfinite(value.detach()).all().item():
                bad_parts.append(name)
        if bad_parts:
            values = ", ".join(
                f"{name}={float(loss_parts[name].detach().cpu().item())}"
                for name in bad_parts
                if torch.is_tensor(loss_parts[name]) and loss_parts[name].numel() == 1
            )
            raise RuntimeError(f"Non-finite loss detected at {context}: {values or bad_parts}")

    def _assert_finite_gradients(self, model, context: str) -> None:
        bad_names = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                bad_names.append(name)
                if len(bad_names) >= 8:
                    break
        if bad_names:
            raise RuntimeError(
                "Non-finite gradient detected at "
                f"{context}; first affected parameters: {', '.join(bad_names)}"
            )

    def _forecast_horizon(self, model, inputs: torch.Tensor, horizon: int) -> torch.Tensor:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        current = inputs
        outputs = []
        for _step in range(horizon):
            step_prediction = model(current)
            if step_prediction.dim() == 4:
                step_prediction = step_prediction.unsqueeze(1)
            if step_prediction.shape[1] != 1:
                raise ValueError(
                    "The model must return exactly one predicted timestep during recursive rollout; "
                    f"got shape {tuple(step_prediction.shape)}."
                )
            outputs.append(step_prediction[:, 0])
            current = torch.cat([current[:, 1:], step_prediction], dim=1)
        return torch.stack(outputs, dim=1)

    def evaluate_model(self, model, criterion, loader, rollout: bool = True) -> dict:
        model.eval()
        running_total_loss = 0.0
        running_data_loss = 0.0
        running_raw_data_loss = 0.0
        running_residual_loss = 0.0
        running_residual_gradient_loss = 0.0
        running_residual_distribution_loss = 0.0
        running_pinn_loss = 0.0
        running_raw_pinn_loss = 0.0
        batch_count = 0
        max_batches = self.config.test_batches

        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                inputs, targets, _target_indexes, parameters, mask, time_deltas = self._prepare_batch(batch)
                if rollout:
                    predictions = self._forecast_horizon(model, inputs, targets.shape[1])
                    loss_targets = targets
                    loss_time_deltas = time_deltas
                else:
                    predictions = model(inputs)
                    loss_targets = targets[:, :1]
                    loss_time_deltas = time_deltas[:, :1] if time_deltas.dim() == 2 else time_deltas[:1]
                loss_parts = criterion.components(
                    prediction=predictions,
                    target=loss_targets,
                    mask=mask,
                    parameters=parameters,
                    input_sequence=inputs,
                    time_deltas=loss_time_deltas,
                )
                running_total_loss += float(loss_parts["total_loss"].item())
                running_data_loss += float(loss_parts["data_loss"].item())
                running_raw_data_loss += float(loss_parts["raw_data_loss"].item())
                running_residual_loss += float(loss_parts["residual_loss"].item())
                running_residual_gradient_loss += float(loss_parts["residual_gradient_loss"].item())
                running_residual_distribution_loss += float(loss_parts["residual_distribution_loss"].item())
                running_pinn_loss += float(loss_parts["pinn_loss"].item())
                running_raw_pinn_loss += float(loss_parts["raw_pinn_loss"].item())
                batch_count += 1
                del inputs, targets, parameters, predictions, loss_parts, mask, time_deltas

        model.train()
        self.cleanup_memory()
        if batch_count == 0:
            return {
                "total_loss": float("inf"),
                "data_loss": float("inf"),
                "raw_data_loss": float("inf"),
                "residual_loss": float("inf"),
                "residual_gradient_loss": float("inf"),
                "residual_distribution_loss": float("inf"),
                "pinn_loss": float("inf"),
                "raw_pinn_loss": float("inf"),
            }
        return {
            "total_loss": running_total_loss / batch_count,
            "data_loss": running_data_loss / batch_count,
            "raw_data_loss": running_raw_data_loss / batch_count,
            "residual_loss": running_residual_loss / batch_count,
            "residual_gradient_loss": running_residual_gradient_loss / batch_count,
            "residual_distribution_loss": running_residual_distribution_loss / batch_count,
            "pinn_loss": running_pinn_loss / batch_count,
            "raw_pinn_loss": running_raw_pinn_loss / batch_count,
        }

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        loss: float,
        train_losses,
        test_losses,
        val_losses,
        filename: str,
    ) -> Path:
        assert self.data is not None
        path = self.config.model_dir / filename
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self._model_state_for_checkpoint(model),
            "loss": float(loss),
            "train_losses": list(train_losses[-self.config.loss_history_limit :]),
            "test_losses": list(test_losses[-self.config.loss_history_limit :]),
            "val_losses": list(val_losses[-self.config.loss_history_limit :]),
            "config": self._config_snapshot(),
            "normalizer": asdict(self.data.normalizer),
            "optimizer_state_dict": None,
        }
        if self.config.save_optimizer_state:
            payload["optimizer_state_dict"] = self._optimizer_state_for_checkpoint(optimizer)

        temp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp_path)
        temp_path.replace(path)
        del payload
        self.cleanup_memory()
        return path

    def _model_state_for_checkpoint(self, model) -> dict:
        state = {}
        for name, tensor in model.state_dict().items():
            value = tensor.detach()
            if self.config.checkpoint_on_cpu:
                value = value.cpu()
            state[name] = value
        return state

    def _optimizer_state_for_checkpoint(self, optimizer) -> dict:
        state = optimizer.state_dict()
        if not self.config.checkpoint_on_cpu:
            return state
        return self._move_tensors_to_cpu(state)

    def _move_tensors_to_cpu(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: self._move_tensors_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_tensors_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_tensors_to_cpu(item) for item in value)
        return value

    def load_model(self, checkpoint_path: Optional[Path] = None) -> PatchwiseRasterForecastModel:
        path = Path(checkpoint_path) if checkpoint_path else self.config.model_dir / "best_lgtm_lager_pinn_gradgate_model.pth"
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        model = self.model or self.build_model()
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        del checkpoint
        self.cleanup_memory()
        return model

    def predict_all(self, model: Optional[PatchwiseRasterForecastModel] = None) -> int:
        if self.data is None:
            self.setup_data()
        assert self.data is not None and self.data.reader is not None and self.data.catalog is not None

        model = model or self.model or self.load_model()
        model.eval()
        output_dir = self.config.output_dir / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "test", "val"):
            (output_dir / split_name).mkdir(parents=True, exist_ok=True)

        saved = 0
        loader = self.data.full_loader(batch_size=1)
        with torch.no_grad():
            saved += self._save_initial_predictions(model, output_dir)
            for batch in tqdm(loader, desc="Predicting"):
                inputs, _targets, target_indexes, _parameters, _mask, _time_deltas = self._prepare_batch(batch)
                prediction = model(inputs)
                prediction = self.data.denormalize(prediction.detach().cpu())

                for step in range(prediction.shape[1]):
                    target_index = int(target_indexes[0, step].item())
                    record = self.data.catalog[target_index]
                    array = prediction[0, step].numpy().astype(np.float32)
                    self._write_prediction(array, record.filename, self._prediction_split_dir(target_index, output_dir))
                    saved += 1
                    del array
                del inputs, prediction, target_indexes

        print(f"Saved {saved} prediction rasters to: {output_dir}")
        self.cleanup_memory()
        return saved

    def _prediction_split_dir(self, target_index: int, base_dir: Path) -> Path:
        train_end = self.config.train_timesteps
        test_end = train_end + self.config.test_timesteps
        val_end = test_end + self.config.val_timesteps
        if target_index < train_end:
            split_dir = base_dir / "train"
        elif target_index < test_end:
            split_dir = base_dir / "test"
        elif target_index < val_end:
            split_dir = base_dir / "val"
        else:
            split_dir = base_dir / "unknown"
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir

    def predict_future(self, model: Optional[PatchwiseRasterForecastModel] = None, steps: Optional[int] = None) -> int:
        if self.data is None:
            self.setup_data()
        assert self.data is not None and self.data.reader is not None and self.data.catalog is not None

        steps = self.config.future_steps if steps is None else int(steps)
        if steps <= 0:
            print("Future extrapolation skipped because future_steps <= 0.")
            return 0

        model = model or self.model or self.load_model()
        model.eval()
        output_dir = self.config.output_dir / "extrapolation"
        output_dir.mkdir(parents=True, exist_ok=True)

        start_index = len(self.data.catalog) - self.config.seq_length
        if start_index < 0:
            raise ValueError("Not enough timesteps to build the future extrapolation input sequence.")

        frames = [
            self.data.reader.read_frame(index)
            for index in range(start_index, len(self.data.catalog))
        ]
        current = torch.stack(frames, dim=0)
        current = self.data.normalizer.normalize(current).unsqueeze(0).to(self.device)
        step_days = self._future_step_days()

        saved = 0
        with torch.no_grad():
            for step in tqdm(range(1, steps + 1), desc=f"Extrapolating {steps} future scenes"):
                step_prediction = model(current)
                if step_prediction.dim() == 4:
                    step_prediction = step_prediction.unsqueeze(1)
                if step_prediction.shape[1] != 1:
                    raise ValueError(
                        "The model must return exactly one predicted timestep during future extrapolation; "
                        f"got shape {tuple(step_prediction.shape)}."
                    )

                denormalized = self.data.denormalize(step_prediction.detach().cpu())
                array = denormalized[0, 0].numpy().astype(np.float32)
                self._write_prediction(array, self._future_prediction_filename(step, step_days), output_dir)
                current = torch.cat([current[:, 1:], step_prediction[:, :1]], dim=1)
                saved += 1
                del denormalized, array, step_prediction

        del frames, current
        print(f"Saved {saved} future extrapolation rasters to: {output_dir}")
        self.cleanup_memory()
        return saved

    def _future_step_days(self) -> int:
        assert self.data is not None and self.data.catalog is not None
        dates = [record.date for record in self.data.catalog.records if record.date is not None]
        deltas = [
            (current - previous).days
            for previous, current in zip(dates, dates[1:])
            if (current - previous).days > 0
        ]
        if deltas:
            return max(int(round(float(np.median(deltas)))), 1)
        return max(int(round(float(self.config.time_step))), 1)

    def _future_prediction_filename(self, step: int, step_days: int) -> str:
        assert self.data is not None and self.data.catalog is not None
        last_record = self.data.catalog[-1]
        if last_record.date is not None:
            future_date = last_record.date + timedelta(days=step_days * step)
            return f"{future_date:%Y%m%d}.tif"
        return f"future_{step:03d}.tif"

    def _save_initial_predictions(self, model, output_dir: Path) -> int:
        assert self.data is not None and self.data.reader is not None and self.data.catalog is not None
        first_frame = self.data.reader.read_frame(0)
        first_sequence = torch.stack([first_frame for _ in range(self.config.seq_length)], dim=0)
        first_sequence = self.data.normalizer.normalize(first_sequence).unsqueeze(0).to(self.device)
        prediction = model(first_sequence)
        prediction = self.data.denormalize(prediction.detach().cpu())
        array = prediction[0, 0].numpy().astype(np.float32)
        saved = 0
        for index in range(min(self.config.seq_length, len(self.data.catalog))):
            self._write_prediction(array, self.data.catalog[index].filename, self._prediction_split_dir(index, output_dir))
            saved += 1
        del first_frame, first_sequence, prediction, array
        return saved

    def _write_prediction(self, array: np.ndarray, source_filename: str, output_dir: Path) -> Path:
        assert self.data is not None and self.data.reader is not None
        output_name = f"{Path(source_filename).stem}_{self.config.prediction_suffix}.tif"
        output_path = output_dir / output_name
        profile = self.data.reader.output_profile(count=array.shape[0], dtype="float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array)
        return output_path

    def evaluate_split_predictions(self, split_name: str, prediction_dir: Optional[Path] = None) -> dict:
        if self.data is None:
            self.setup_data()
        assert self.data is not None and self.data.reader is not None and self.data.catalog is not None

        ranges = {
            "train": (0, self.config.train_timesteps),
            "test": (self.config.train_timesteps, self.config.train_timesteps + self.config.test_timesteps),
            "val": (
                self.config.train_timesteps + self.config.test_timesteps,
                self.config.train_timesteps + self.config.test_timesteps + self.config.val_timesteps,
            ),
        }
        if split_name not in ranges:
            raise ValueError(f"split_name must be one of {sorted(ranges)}, got {split_name!r}")

        base_prediction_dir = prediction_dir or self.config.output_dir / "predictions"
        split_prediction_dir = base_prediction_dir / split_name
        if not split_prediction_dir.exists():
            raise FileNotFoundError(f"Prediction directory not found: {split_prediction_dir}")

        start, end = ranges[split_name]
        rows = []
        for target_index in range(start, end):
            record = self.data.catalog[target_index]
            prediction_path = split_prediction_dir / f"{Path(record.filename).stem}_{self.config.prediction_suffix}.tif"
            if not prediction_path.exists():
                raise FileNotFoundError(f"Missing prediction for {split_name} raster {record.filename}: {prediction_path}")

            with rasterio.open(prediction_path) as src:
                prediction = src.read(1).astype(np.float32)
            target = self.data.reader.read_frame(target_index).squeeze(0).numpy().astype(np.float32)
            rows.append({"filename": record.filename, **self._metric_row(prediction, target)})

        summary = self._summarize_metrics(rows)
        self._save_metrics(rows, summary, split_name)
        print(f"{split_name.capitalize()} prediction metrics")
        print(f"  Average RMSE: {summary['rmse']:.6f}")
        print(f"  MAE: {summary['mae']:.6f}")
        print(f"  R2: {summary['r2']:.6f}")
        print(f"  SSIM: {summary['ssim']:.6f}")
        print(f"  IoU: {summary['iou']:.6f}")
        return summary

    def evaluate_test_predictions(self, prediction_dir: Optional[Path] = None) -> dict:
        return self.evaluate_split_predictions("test", prediction_dir=prediction_dir)

    def _metric_row(self, prediction: np.ndarray, target: np.ndarray) -> dict:
        valid = np.isfinite(prediction) & np.isfinite(target)
        if not np.any(valid):
            return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "ssim": np.nan, "iou": np.nan}

        pred = prediction[valid].astype(np.float64)
        truth = target[valid].astype(np.float64)
        diff = pred - truth
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        ss_res = float(np.sum(diff**2))
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        data_range = float(np.nanmax(target) - np.nanmin(target))
        if data_range <= 0:
            ssim_value = np.nan
        else:
            ssim_value = float(ssim(target, prediction, data_range=data_range))

        threshold = float(getattr(self.config, "metric_iou_threshold", 0.0))
        pred_mask = np.isfinite(prediction) & (prediction < threshold)
        target_mask = np.isfinite(target) & (target < threshold)
        union = np.logical_or(pred_mask, target_mask).sum()
        intersection = np.logical_and(pred_mask, target_mask).sum()
        iou = float(intersection / union) if union > 0 else np.nan
        return {"rmse": rmse, "mae": mae, "r2": r2, "ssim": ssim_value, "iou": iou}

    def _summarize_metrics(self, rows) -> dict:
        metric_names = ("rmse", "mae", "r2", "ssim", "iou")
        return {
            name: float(np.nanmean([row[name] for row in rows]))
            for name in metric_names
        }

    def _save_metrics(self, rows, summary: dict, split_name: str) -> None:
        metrics_path = self.config.output_dir / f"{split_name}_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["filename", "rmse", "mae", "r2", "ssim", "iou"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary_path = self.config.output_dir / f"{split_name}_metrics_summary.txt"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(f"{split_name.capitalize()} prediction metrics\n")
            handle.write(f"Average RMSE: {summary['rmse']:.6f}\n")
            handle.write(f"MAE: {summary['mae']:.6f}\n")
            handle.write(f"R2: {summary['r2']:.6f}\n")
            handle.write(f"SSIM: {summary['ssim']:.6f}\n")
            handle.write(f"IoU: {summary['iou']:.6f}\n")
            handle.write(f"IoU threshold: prediction/target < {getattr(self.config, 'metric_iou_threshold', 0.0)}\n")
        print(f"Saved {split_name} metrics: {metrics_path}")
        print(f"Saved {split_name} metrics summary: {summary_path}")

    def run(self) -> PatchwiseRasterForecastModel:
        model = self.train_model()
        if self.config.save_predictions:
            self.predict_all(model)
            for split_name in ("train", "test", "val"):
                self.evaluate_split_predictions(split_name)
            self.predict_future(model)
        return model

    def smoke_test(self) -> None:
        if self.data is None:
            self.setup_data()
        model = self.model or self.build_model()
        criterion = build_loss(self.config, normalizer=self.data.normalizer).to(self.device)
        batch = next(iter(self.data.train_loader()))
        inputs, targets, _target_indexes, parameters, mask, time_deltas = self._prepare_batch(batch)
        model.train()
        predictions = model(inputs)
        loss_parts = criterion.components(
            prediction=predictions,
            target=targets,
            mask=mask,
            parameters=parameters,
            input_sequence=inputs,
            time_deltas=time_deltas,
        )
        loss = loss_parts["total_loss"]
        loss.backward()
        print(
            "Smoke test passed: "
            f"inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}, "
            f"predictions={tuple(predictions.shape)}, "
            f"total_loss={float(loss_parts['total_loss'].item()):.6f}, "
            f"data_loss={float(loss_parts['data_loss'].item()):.6f}, "
            f"residual_loss={float(loss_parts['residual_loss'].item()):.6f}, "
            f"residual_gradient_loss={float(loss_parts['residual_gradient_loss'].item()):.6f}, "
            f"PINN_loss={float(loss_parts['pinn_loss'].item()):.6f}, "
            f"raw_PINN_loss={float(loss_parts['raw_pinn_loss'].item()):.6f}"
        )
        test_batch = next(iter(self.data.test_loader()))
        test_inputs, test_targets, _test_target_indexes, _test_parameters, _test_mask, _test_time_deltas = self._prepare_batch(
            test_batch
        )
        val_batch = next(iter(self.data.val_loader()))
        val_inputs, val_targets, _val_target_indexes, _val_parameters, _val_mask, _val_time_deltas = self._prepare_batch(
            val_batch
        )
        with torch.no_grad():
            test_predictions = model(test_inputs)
            val_predictions = model(val_inputs)
        print(
            "Sliding-window test check: "
            f"inputs={tuple(test_inputs.shape)}, targets={tuple(test_targets.shape)}, "
            f"predictions={tuple(test_predictions.shape)}"
        )
        print(
            "Sliding-window validation check: "
            f"inputs={tuple(val_inputs.shape)}, targets={tuple(val_targets.shape)}, "
            f"predictions={tuple(val_predictions.shape)}"
        )
        del criterion, inputs, targets, parameters, mask, time_deltas, predictions, loss, loss_parts
        del test_inputs, test_targets, _test_parameters, _test_mask, _test_time_deltas, test_predictions
        del val_inputs, val_targets, _val_parameters, _val_mask, _val_time_deltas, val_predictions
        self.cleanup_memory()

    def _prepare_batch(self, batch):
        if len(batch) != 6:
            raise ValueError(f"Expected batch with 6 items, got {len(batch)}")
        inputs, targets, target_indexes, parameters, mask, time_deltas = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        time_deltas = time_deltas.to(self.device, non_blocking=True)
        parameters = {
            name: value.to(self.device, non_blocking=True)
            for name, value in parameters.items()
        }
        return inputs, targets, target_indexes, parameters, mask, time_deltas

    def _config_snapshot(self) -> dict:
        snapshot = asdict(self.config)
        for key, value in list(snapshot.items()):
            if isinstance(value, Path):
                try:
                    snapshot[key] = str(value.resolve().relative_to(Path(__file__).resolve().parent))
                except ValueError:
                    snapshot[key] = str(value)
        return snapshot


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LGTM + 1987 soft-soil-3 dominant consolidation PINN raster forecasting.")
    parser.add_argument("--data-path", type=Path, default=None, help="Directory containing 8-digit tif time series.")
    parser.add_argument("--parameter-dir", type=Path, default=None, help="Directory containing flat 1987 Parameter/*.tif rasters.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size.")
    parser.add_argument("--patch-size", type=int, default=None, help="Square patch size override.")
    parser.add_argument("--patch-stride", type=int, default=None, help="Square patch stride override.")
    parser.add_argument("--patch-stage-size", type=int, default=None, help="Patch chunks processed per stage.")
    parser.add_argument("--random-seed", type=int, default=None, help="Spatial split random seed.")
    parser.add_argument("--resize-height", type=int, default=None, help="Optional smoke/small-run resize height.")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional smoke/small-run resize width.")
    parser.add_argument("--iou-threshold", type=float, default=None, help="Threshold for settlement-mask IoU.")
    parser.add_argument("--future-steps", type=int, default=None, help="Number of future scenes to extrapolate.")
    parser.add_argument("--no-predict", action="store_true", help="Train only; do not write prediction rasters.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one forward/backward pass only.")
    return parser


def explicit_config() -> LGTMConfig:
    return LGTMConfig(**EXPLICIT_LGTM_CONFIG)


def config_from_args(args: argparse.Namespace) -> LGTMConfig:
    kwargs = dict(EXPLICIT_LGTM_CONFIG)
    if args.data_path is not None:
        kwargs["data_path"] = args.data_path
    if args.parameter_dir is not None:
        kwargs["parameter_raster_dir"] = args.parameter_dir
    if args.epochs is not None:
        kwargs["num_epochs"] = args.epochs
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.patch_size is not None:
        kwargs["patch_size"] = (args.patch_size, args.patch_size)
    if args.patch_stride is not None:
        kwargs["patch_stride"] = (args.patch_stride, args.patch_stride)
    if args.patch_stage_size is not None:
        kwargs["patch_stage_size"] = args.patch_stage_size
    if args.random_seed is not None:
        kwargs["random_seed"] = args.random_seed
    if args.resize_height is not None or args.resize_width is not None:
        if args.resize_height is None or args.resize_width is None:
            raise ValueError("--resize-height and --resize-width must be supplied together.")
        kwargs["resize_shape"] = (args.resize_height, args.resize_width)
    if args.iou_threshold is not None:
        kwargs["metric_iou_threshold"] = args.iou_threshold
    if args.future_steps is not None:
        kwargs["future_steps"] = args.future_steps
    if args.no_predict:
        kwargs["save_predictions"] = False
    return LGTMConfig(**kwargs)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    pipeline = LGTM(config)
    try:
        if args.smoke_test:
            pipeline.smoke_test()
        else:
            pipeline.run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
