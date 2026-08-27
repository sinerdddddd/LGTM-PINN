from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _match_spatial(parameter: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Resize/crop parameter raster tensor to match [B, L, H, W]."""
    if parameter.dim() == 5 and parameter.shape[1] == 1:
        parameter = parameter[:, 0]
    if parameter.dim() == 3:
        parameter = parameter.unsqueeze(0)
    if parameter.dim() != 4:
        raise ValueError(f"Parameter tensor must be [B,L,H,W] or [L,H,W], got {tuple(parameter.shape)}")
    if parameter.shape[-2:] != reference.shape[-2:]:
        parameter = F.interpolate(parameter, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    if parameter.shape[0] == 1 and reference.shape[0] > 1:
        parameter = parameter.expand(reference.shape[0], -1, -1, -1)
    return parameter.to(device=reference.device, dtype=reference.dtype)


def _to_physical(tensor: torch.Tensor, data_min: float, data_scale: float) -> torch.Tensor:
    return (tensor + 1.0) * data_scale + data_min


class LGTMPINNLoss(nn.Module):
    """Data loss plus a physically constrained consolidation residual.

    For the 1987 soft-soil-3 setup the model still predicts total surface
    displacement, while the PINN term constrains the dominant deep soft-soil
    layer using a depth-integrated first-order consolidation residual:

        r3 = w3 * dS/dt - lambda3 * (Sinf3 - w3 * S)
        lambda3 = pi^2 * cv3 / (4 * Hdr3^2)

    The displacement rasters are negative for subsidence; settlement_sign=-1
    converts them to positive settlement before evaluating the residual.
    """

    def __init__(
        self,
        loss_name: str = "huber",
        delta: float = 1.0,
        data_loss_weight: float = 1.0,
        residual_loss_weight: float = 0.5,
        residual_gradient_loss_weight: float = 0.1,
        residual_distribution_loss_weight: float = 0.01,
        residual_scale_min: float = 1e-3,
        use_pinn_loss: bool = True,
        pinn_loss_weight: float = 0.1,
        report_raw_pinn_loss: bool = False,
        pinn_mode: str = "soft3_dominant",
        time_step: float = 1.0,
        depth_step: float = 5.0,
        stress_scale_kpa: float = 1.0,
        data_min: float = 0.0,
        data_scale: float = 1.0,
        cv_is_log10: bool = True,
        cv_cm2s_to_m2day: bool = True,
        cv_cm2year_to_m2day: bool = False,
        settlement_sign: float = -1.0,
        pinn_residual_scale_mm_per_day: float = 1.0,
        soft3_drainage: str = "single",
    ) -> None:
        super().__init__()
        self.loss_name = loss_name.lower()
        self.delta = delta
        self.data_loss_weight = float(data_loss_weight)
        self.residual_loss_weight = float(residual_loss_weight)
        self.residual_gradient_loss_weight = float(residual_gradient_loss_weight)
        self.residual_distribution_loss_weight = float(residual_distribution_loss_weight)
        self.residual_scale_min = float(residual_scale_min)
        self.use_pinn_loss = use_pinn_loss
        self.pinn_loss_weight = float(pinn_loss_weight)
        self.report_raw_pinn_loss = bool(report_raw_pinn_loss)
        self.pinn_mode = pinn_mode
        self.time_step = float(time_step)
        self.depth_step = float(depth_step)
        self.stress_scale_kpa = float(stress_scale_kpa)
        self.data_min = float(data_min)
        self.data_scale = float(data_scale)
        self.cv_is_log10 = bool(cv_is_log10)
        self.cv_cm2s_to_m2day = bool(cv_cm2s_to_m2day)
        self.cv_cm2year_to_m2day = bool(cv_cm2year_to_m2day)
        self.settlement_sign = float(settlement_sign)
        self.pinn_residual_scale_mm_per_day = float(pinn_residual_scale_mm_per_day)
        self.soft3_drainage = soft3_drainage
        if self.loss_name not in {"huber", "mse", "mae"}:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        parameters: Optional[Dict[str, torch.Tensor]] = None,
        input_sequence: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.components(
            prediction=prediction,
            target=target,
            mask=mask,
            parameters=parameters,
            input_sequence=input_sequence,
            time_deltas=time_deltas,
        )["total_loss"]

    def components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        parameters: Optional[Dict[str, torch.Tensor]] = None,
        input_sequence: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}")

        raw_data_loss = self._data_loss(prediction, target, mask)
        weighted_data_loss = self.data_loss_weight * raw_data_loss
        zero = raw_data_loss.new_tensor(0.0)
        should_compute_physics = (
            (self.use_pinn_loss or self.report_raw_pinn_loss)
            and (self.pinn_loss_weight > 0 or self.report_raw_pinn_loss)
            and parameters is not None
            and input_sequence is not None
        )
        if not should_compute_physics:
            return {
                "total_loss": weighted_data_loss,
                "data_loss": weighted_data_loss,
                "raw_data_loss": raw_data_loss,
                "residual_loss": zero,
                "raw_residual_loss": zero,
                "residual_gradient_loss": zero,
                "raw_residual_gradient_loss": zero,
                "residual_distribution_loss": zero,
                "raw_residual_distribution_loss": zero,
                "pinn_loss": zero,
                "raw_pinn_loss": zero,
            }

        if self.pinn_loss_weight > 0:
            prediction_physical = _to_physical(prediction, self.data_min, self.data_scale)
            input_physical = _to_physical(input_sequence, self.data_min, self.data_scale)
            physics_loss = self._physics_loss(
                prediction=prediction_physical,
                parameters=parameters,
                input_sequence=input_physical,
                mask=mask,
                time_deltas=time_deltas,
            )
            weighted_physics_loss = self.pinn_loss_weight * physics_loss
            total_loss = weighted_data_loss + weighted_physics_loss
        else:
            with torch.no_grad():
                prediction_physical = _to_physical(prediction, self.data_min, self.data_scale)
                input_physical = _to_physical(input_sequence, self.data_min, self.data_scale)
                physics_loss = self._physics_loss(
                    prediction=prediction_physical,
                    parameters=parameters,
                    input_sequence=input_physical,
                    mask=mask,
                    time_deltas=time_deltas,
                )
            weighted_physics_loss = zero
            total_loss = weighted_data_loss
        return {
            "total_loss": total_loss,
            "data_loss": weighted_data_loss,
            "raw_data_loss": raw_data_loss,
            "residual_loss": zero,
            "raw_residual_loss": zero,
            "residual_gradient_loss": zero,
            "raw_residual_gradient_loss": zero,
            "residual_distribution_loss": zero,
            "raw_residual_distribution_loss": zero,
            "pinn_loss": weighted_physics_loss,
            "raw_pinn_loss": physics_loss,
        }

    def _data_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.loss_name == "huber":
            loss_map = F.huber_loss(prediction, target, delta=self.delta, reduction="none")
        elif self.loss_name == "mse":
            loss_map = F.mse_loss(prediction, target, reduction="none")
        else:
            loss_map = F.l1_loss(prediction, target, reduction="none")

        if mask is None:
            return loss_map.mean()

        weight = mask.to(dtype=loss_map.dtype, device=loss_map.device)
        while weight.dim() < loss_map.dim():
            weight = weight.unsqueeze(1)
        weight = weight.expand_as(loss_map)
        return (loss_map * weight).sum() / weight.sum().clamp_min(1.0)

    def _residual_components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        input_sequence: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = prediction.new_tensor(0.0)
        if input_sequence is None or prediction.shape[1] != 1 or target.shape[1] != 1:
            return {
                "residual_loss": zero,
                "residual_gradient_loss": zero,
                "residual_distribution_loss": zero,
            }

        previous = input_sequence[:, -1]
        if previous.dim() == 3:
            previous = previous.unsqueeze(1)
        prediction_residual = prediction[:, 0] - previous
        target_residual = target[:, 0] - previous

        scale = target_residual.detach().abs().mean().clamp_min(self.residual_scale_min)
        normalized_prediction_residual = prediction_residual / scale
        normalized_target_residual = target_residual / scale

        residual_loss = self._residual_loss(
            normalized_prediction_residual,
            normalized_target_residual,
            mask,
        )
        gradient_loss = self._gradient_loss(
            normalized_prediction_residual,
            normalized_target_residual,
            mask,
        )
        distribution_loss = self._distribution_loss(
            normalized_prediction_residual,
            normalized_target_residual,
            mask,
        )
        return {
            "residual_loss": residual_loss,
            "residual_gradient_loss": gradient_loss,
            "residual_distribution_loss": distribution_loss,
        }

    def _residual_loss(
        self,
        prediction_residual: torch.Tensor,
        target_residual: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.loss_name == "mse":
            loss_map = F.mse_loss(prediction_residual, target_residual, reduction="none")
        elif self.loss_name == "mae":
            loss_map = F.l1_loss(prediction_residual, target_residual, reduction="none")
        else:
            loss_map = F.huber_loss(prediction_residual, target_residual, delta=self.delta, reduction="none")
        return self._masked_mean(loss_map, mask)

    def _gradient_loss(
        self,
        prediction_residual: torch.Tensor,
        target_residual: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pred_dy = prediction_residual[..., 1:, :] - prediction_residual[..., :-1, :]
        true_dy = target_residual[..., 1:, :] - target_residual[..., :-1, :]
        pred_dx = prediction_residual[..., :, 1:] - prediction_residual[..., :, :-1]
        true_dx = target_residual[..., :, 1:] - target_residual[..., :, :-1]
        if mask is None:
            return F.l1_loss(pred_dy, true_dy) + F.l1_loss(pred_dx, true_dx)

        mask_y = mask[..., 1:, :] * mask[..., :-1, :]
        mask_x = mask[..., :, 1:] * mask[..., :, :-1]
        weight_y = self._expand_mask(mask_y, pred_dy)
        weight_x = self._expand_mask(mask_x, pred_dx)
        loss_y = (F.l1_loss(pred_dy, true_dy, reduction="none") * weight_y).sum()
        loss_y = loss_y / weight_y.sum().clamp_min(1.0)
        loss_x = (F.l1_loss(pred_dx, true_dx, reduction="none") * weight_x).sum()
        loss_x = loss_x / weight_x.sum().clamp_min(1.0)
        return loss_y + loss_x

    def _distribution_loss(
        self,
        prediction_residual: torch.Tensor,
        target_residual: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            pred_mean = prediction_residual.mean(dim=(-2, -1))
            true_mean = target_residual.mean(dim=(-2, -1))
            pred_var = (prediction_residual - pred_mean[..., None, None]).square().mean(dim=(-2, -1))
            true_var = (target_residual - true_mean[..., None, None]).square().mean(dim=(-2, -1))
        else:
            weight = self._expand_mask(mask, prediction_residual)
            denom = weight.sum(dim=(-2, -1)).clamp_min(1.0)
            pred_mean = (prediction_residual * weight).sum(dim=(-2, -1)) / denom
            true_mean = (target_residual * weight).sum(dim=(-2, -1)) / denom
            pred_var = ((prediction_residual - pred_mean[..., None, None]) ** 2 * weight).sum(dim=(-2, -1)) / denom
            true_var = ((target_residual - true_mean[..., None, None]) ** 2 * weight).sum(dim=(-2, -1)) / denom
        return F.l1_loss(pred_mean, true_mean) + F.l1_loss(pred_var, true_var)

    def _masked_mean(self, loss_map: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return loss_map.mean()
        weight = self._expand_mask(mask, loss_map)
        return (loss_map * weight).sum() / weight.sum().clamp_min(1.0)

    def _expand_mask(self, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        weight = mask.to(dtype=reference.dtype, device=reference.device)
        while weight.dim() < reference.dim():
            weight = weight.unsqueeze(1)
        return weight.expand_as(reference)

    def _physics_loss(
        self,
        prediction: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        input_sequence: torch.Tensor,
        mask: Optional[torch.Tensor],
        time_deltas: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.pinn_mode == "soft3_dominant":
            return self._soft3_dominant_loss(prediction, parameters, input_sequence, mask, time_deltas)
        if self.pinn_mode == "depth_integrated":
            return self._depth_integrated_loss(prediction, parameters, input_sequence, mask, time_deltas)
        return self._terzaghi_1d_loss(prediction, parameters, input_sequence, mask, time_deltas)

    def _cv_tensor(self, cv: torch.Tensor) -> torch.Tensor:
        if self.cv_is_log10:
            cv = torch.clamp(cv, min=-10.0, max=10.0)
            cv = torch.pow(torch.tensor(10.0, dtype=cv.dtype, device=cv.device), cv)
        if self.cv_cm2s_to_m2day:
            cv = cv * 86400.0 / 10000.0
        if self.cv_cm2year_to_m2day:
            # The 1987 workbook computes cv = k*(1+e)/(0.0001*a) from
            # k(cm/year), so cv is cm^2/year. Convert to m^2/day.
            cv = cv / (10000.0 * 365.0)
        return cv

    def _soft3_dominant_loss(
        self,
        prediction: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        input_sequence: torch.Tensor,
        mask: Optional[torch.Tensor],
        time_deltas: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cv = parameters.get("cv_mean")
        sinf = parameters.get("Sinf_sum_mm")
        weight = parameters.get("contribution_weight_soft_only")
        hdr_name = "Hdr_double_m" if self.soft3_drainage == "double" else "Hdr_single_m"
        hdr = parameters.get(hdr_name)
        if cv is None or sinf is None or weight is None or hdr is None:
            return prediction.new_tensor(0.0)

        ref = prediction[:, 0]
        cv = self._cv_tensor(_match_spatial(cv, ref)).clamp_min(0.0)
        sinf = _match_spatial(sinf, ref).clamp_min(0.0)
        weight = _match_spatial(weight, ref).clamp(0.0, 1.0)
        hdr = _match_spatial(hdr, ref).clamp_min(1e-6)
        lam = (torch.pi ** 2) * cv / (4.0 * hdr.square())

        previous = input_sequence[:, -1]
        if previous.dim() == 3:
            previous = previous.unsqueeze(1)

        losses = []
        scale = max(self.pinn_residual_scale_mm_per_day, 1e-12)
        for step in range(prediction.shape[1]):
            current = prediction[:, step]
            last = previous if step == 0 else prediction[:, step - 1]
            delta_t = self._time_delta(step, prediction, time_deltas)

            current_settlement = self.settlement_sign * current
            last_settlement = self.settlement_sign * last
            dS_dt = (current_settlement - last_settlement) / delta_t
            layer_settlement = weight * current_settlement
            residual = weight * dS_dt - lam * (sinf - layer_settlement)
            residual = residual / scale
            losses.append(self._masked_mean_square(residual, mask))
        return torch.stack(losses).mean()

    def _terzaghi_1d_loss(
        self,
        prediction: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        input_sequence: torch.Tensor,
        mask: Optional[torch.Tensor],
        time_deltas: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cv = parameters.get("Cv_cm2s_lg10")
        mv = parameters.get("mv_0_1_0_2_MPa_inv")
        thickness = parameters.get("LayerThickness_m")
        if cv is None or mv is None or thickness is None:
            return prediction.new_tensor(0.0)

        ref = prediction[:, 0]
        cv = self._cv_tensor(_match_spatial(cv, ref))
        mv = _match_spatial(mv, ref).clamp_min(0.0)
        thickness = _match_spatial(thickness, ref).clamp_min(0.0)

        weights = mv * thickness
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        previous_surface = input_sequence[:, -1]
        if previous_surface.dim() == 3:
            previous_surface = previous_surface.unsqueeze(1)

        losses = []
        for step in range(prediction.shape[1]):
            current_surface = prediction[:, step]
            if step == 0:
                last_surface = previous_surface
            else:
                last_surface = prediction[:, step - 1]

            delta_t = self._time_delta(step, prediction, time_deltas)
            current_layers = current_surface * weights
            last_layers = last_surface * weights
            ds_dt = (current_layers - last_layers) / delta_t

            padded = F.pad(current_layers, (0, 0, 0, 0, 1, 1), mode="replicate")
            d2s_dz2 = (padded[:, 2:] - 2.0 * padded[:, 1:-1] + padded[:, :-2]) / (self.depth_step**2)
            residual = ds_dt - cv * d2s_dz2
            losses.append(self._masked_mean_square(residual, mask))
        return torch.stack(losses).mean()

    def _depth_integrated_loss(
        self,
        prediction: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        input_sequence: torch.Tensor,
        mask: Optional[torch.Tensor],
        time_deltas: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cv = parameters.get("Cv_cm2s_lg10")
        mv = parameters.get("mv_0_1_0_2_MPa_inv")
        thickness = parameters.get("LayerThickness_m")
        if cv is None or mv is None:
            return prediction.new_tensor(0.0)

        ref = prediction[:, 0]
        cv = self._cv_tensor(_match_spatial(cv, ref))
        mv = _match_spatial(mv, ref)
        if thickness is not None:
            thickness = _match_spatial(thickness, ref).clamp_min(0.0)
            denom = thickness.sum(dim=1, keepdim=True).clamp_min(1e-6)
            cv_eff = (cv * thickness).sum(dim=1, keepdim=True) / denom
            mv_eff = (mv * thickness).sum(dim=1, keepdim=True) / denom
        else:
            cv_eff = cv.mean(dim=1, keepdim=True)
            mv_eff = mv.mean(dim=1, keepdim=True)

        previous = input_sequence[:, -1]
        if previous.dim() == 3:
            previous = previous.unsqueeze(1)

        losses = []
        for step in range(prediction.shape[1]):
            current = prediction[:, step]
            last = previous if step == 0 else prediction[:, step - 1]
            delta_t = self._time_delta(step, prediction, time_deltas)
            ds_dt = (current - last) / delta_t
            expected_rate = cv_eff * mv_eff * self.stress_scale_kpa
            residual = ds_dt - expected_rate.expand_as(ds_dt)
            losses.append(self._masked_mean_square(residual, mask))
        return torch.stack(losses).mean()

    def _time_delta(self, step: int, reference: torch.Tensor, time_deltas: Optional[torch.Tensor]) -> torch.Tensor:
        if time_deltas is None:
            return reference.new_tensor(self.time_step)
        delta = time_deltas[:, step] if time_deltas.dim() == 2 else time_deltas[step]
        while delta.dim() < reference[:, step].dim():
            delta = delta.view(*delta.shape, 1)
        return delta.to(device=reference.device, dtype=reference.dtype).clamp_min(1e-6)

    def _masked_mean_square(self, residual: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        square = residual.square()
        if mask is None:
            return square.mean()
        weight = mask.to(device=square.device, dtype=square.dtype)
        while weight.dim() < square.dim():
            weight = weight.unsqueeze(1)
        weight = weight.expand_as(square)
        return (square * weight).sum() / weight.sum().clamp_min(1.0)


def build_loss(config, normalizer=None) -> LGTMPINNLoss:
    data_min = normalizer.data_min if normalizer is not None else 0.0
    data_scale = normalizer.data_scale if normalizer is not None else 1.0
    return LGTMPINNLoss(
        loss_name=config.loss_name,
        delta=config.huber_delta,
        data_loss_weight=config.data_loss_weight,
        residual_loss_weight=getattr(config, "residual_loss_weight", 0.0),
        residual_gradient_loss_weight=getattr(config, "residual_gradient_loss_weight", 0.0),
        residual_distribution_loss_weight=getattr(config, "residual_distribution_loss_weight", 0.0),
        residual_scale_min=getattr(config, "residual_scale_min", 1e-3),
        use_pinn_loss=config.use_pinn_loss,
        pinn_loss_weight=config.pinn_loss_weight,
        report_raw_pinn_loss=getattr(config, "report_raw_pinn_loss", False),
        pinn_mode=config.pinn_mode,
        time_step=config.time_step,
        depth_step=config.depth_step,
        stress_scale_kpa=config.stress_scale_kpa,
        data_min=data_min,
        data_scale=data_scale,
        cv_is_log10=config.cv_is_log10,
        cv_cm2s_to_m2day=config.cv_cm2s_to_m2day,
        cv_cm2year_to_m2day=getattr(config, "cv_cm2year_to_m2day", False),
        settlement_sign=getattr(config, "settlement_sign", -1.0),
        pinn_residual_scale_mm_per_day=getattr(config, "pinn_residual_scale_mm_per_day", 1.0),
        soft3_drainage=getattr(config, "soft3_drainage", "single"),
    )
