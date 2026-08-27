# LGTM_PINN_LAGERNEWTIN 梯度角控制版

本目录是 `LGTM_PINN_LAGERNEWT` 的梯度角反向控制 Data-PINN 版本。它保留大容量残差 Informer 和 U-Net，以及软土 3 主控 PINN 方程：

```text
r3 = w3 * dS/dt - lambda3 * (Sinf3 - w3 * S)
lambda3 = pi^2 * cv3 / (4 * Hdr3^2)
```

新增内容是：用 DataLoss 与 PINNloss 的梯度余弦相似度控制下一轮 PINN 权重。前 5 个 epoch 只诊断不控制；之后根据 `cos_ema` 计算 `gradient_gate`，并令：

```text
next_pinn_weight = EMA(adaptive_base_weight * gradient_gate)
```

详细方案、参数、日志字段和量纲说明见 `README_PINN.md`。

运行环境来自：

```text
E:\Project\LGTM_PLUS\LS2\AGENT.read
```

运行：

```powershell
& "E:\Pythonevn\yolov8training\python.exe" LGTM.py
```

快速检查：

```powershell
& "E:\Pythonevn\yolov8training\python.exe" LGTM.py --smoke-test --resize-height 64 --resize-width 64 --no-predict
```
