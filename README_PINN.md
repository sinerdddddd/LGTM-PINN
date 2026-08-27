# LGTM_PINN_LAGERNEWTIN: gradient-angle controlled Data-PINN

本目录由 `LGTM_PINN_LAGERNEWT` 复制而来，是大模型 Data-PINN 的梯度角反向控制版本。代码和必要数据已经包含：

```text
TimeSeriesData_2/
Parameter/
LGTM.py
config.py
data_input.py
losses.py
models.py
```

旧实验输出、旧模型权重和 `__pycache__` 没有复制到本目录，新的训练结果会写入本目录下的 `outputs/` 和 `models/`。

## 模型容量

```text
informer_d_model = 64
informer_n_heads = 4
informer_e_layers = 2
informer_d_ff = 128
unet_features = (32, 64, 128, 256)
prediction_suffix = lgtm_lager_pinn_gradgate
checkpoint = models/best_lgtm_lager_pinn_gradgate_model.pth
```

## PINN 方程

本版本仍使用软土 3 主控的一阶固结型残差。模型预测的是地表位移 `S(t)`，InSAR 沉降为负值，因此 PINN 中先转换为正沉降：

```text
settlement = -displacement
```

微分方程残差为：

```text
r3 = w3 * dS/dt - lambda3 * (Sinf3 - w3 * S)
lambda3 = pi^2 * cv3 / (4 * Hdr3^2)
```

其中：

```text
Sinf3 = 软土 3 最终固结沉降潜势，单位 mm
w3    = 软土 3 沉降贡献权重
Hdr3  = 排水路径，单位 m
cv3   = 固结系数
```

1987 参数表中 `cv = k * (1 + e) / (0.0001 * a)`，`k` 使用 `cm/year`，因此代码将 `cv_mean` 按 `cm^2/year` 读取并转换为 `m^2/day`：

```text
cv_m2_day = cv_cm2_year / (10000 * 365)
```

所以 `lambda3` 的单位是 `1/day`，与按日期差计算的 `time_deltas` 一致，残差单位为 `mm/day`。

## 梯度角反向控制 PINN 权重

旧版大模型只使用 loss 幅值自适应权重：

```text
lambda_base = target_ratio * DataLoss / PINNloss
```

本版本保留该方案作为基础权重，同时引入 DataLoss 与 PINNloss 的梯度角诊断来控制下一轮 PINN 权重。训练流程为：

```text
1. 第 e 轮使用上一轮得到的 lambda_PINN 训练。
2. 第 e 轮结束后，计算 grad(DataLoss) 与 grad(PINNloss) 的余弦相似度 cos(theta)。
3. 对 cos(theta) 做 EMA 平滑，得到 cos_ema。
4. 根据 cos_ema 得到 gradient_gate。
5. 用 lambda_base * gradient_gate 生成第 e+1 轮的 PINN 权重。
```

权重更新形式为：

```text
lambda_base = 原 normalized adaptive PINN weighting 得到的权重
cos_ema     = beta_g * cos_ema_prev + (1 - beta_g) * cos(theta)
lambda_next = EMA(lambda_base * gradient_gate)
```

默认参数为：

```text
use_gradient_angle_pinn_control = True
gradient_angle_control_warmup_epochs = 5
gradient_angle_cosine_ema_beta = 0.8
gradient_gate_negative_cosine = -0.3
gradient_gate_positive_cosine = 0.3
gradient_gate_neutral_value = 0.5
gradient_gate_min_value = 0.05
```

闸门函数为软分段：

```text
cos_ema >=  0.3      -> gradient_gate = 1.0
0 <= cos_ema < 0.3   -> gradient_gate 从 0.5 线性增加到 1.0
-0.3 < cos_ema < 0   -> gradient_gate 从 0.05 线性增加到 0.5
cos_ema <= -0.3      -> gradient_gate = 0.05
```

前 5 个 epoch 为 warm-up，只记录梯度角，不使用梯度角控制权重；从第 6 个 epoch 开始，梯度闸门反作用于下一轮 `PINNloss` 权重。采用软闸门而不是硬关闭，是为了避免钻孔参数插值误差导致 PINN 约束被完全移除。

## 日志字段

`outputs/training_epoch_metrics.csv` 新增字段：

```text
adaptive_base_weight
adaptive_base_raw_weight
gradient_gate
gradient_cosine_ema
gradient_control_active
```

其中 `adaptive_base_weight` 是旧版 loss 幅值自适应权重，`gradient_gate` 是梯度角闸门，`next_pinn_weight` 是真正用于下一轮训练的最终权重。

`outputs/gradient_conflict_diagnostics.csv` 也保留每轮的：

```text
grad_cosine_data_pinn
gradient_angle_degrees
gradient_conflict_fraction
active_pinn_weight
next_pinn_weight
gradient_gate
gradient_cosine_ema
```

## 运行环境

测试和运行环境来自：

```text
E:\Project\LGTM_PLUS\LS2\AGENT.read
```

其中 Python 路径为：

```powershell
E:\Pythonevn\yolov8training\python.exe
```

完整训练：

```powershell
& "E:\Pythonevn\yolov8training\python.exe" LGTM.py
```

快速 smoke test：

```powershell
& "E:\Pythonevn\yolov8training\python.exe" LGTM.py --smoke-test --resize-height 64 --resize-width 64 --no-predict
```
