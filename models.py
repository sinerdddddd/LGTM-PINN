from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_pair(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("Expected an int or a pair.")
    return int(value[0]), int(value[1])


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()[:2]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.pe(positions)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        self.token_conv = nn.Linear(c_in, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.token_conv(x))


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class SimplifiedProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        out = self.dropout(self.proj(attn_out))
        return out, None


class SimplifiedEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2, d_ff: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = SimplifiedProbSparseAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask=None):
        attn_out, _ = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, None


class SimpleInformerP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        pred_length: int,
        d_model: int = 32,
        n_heads: int = 2,
        e_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.pred_length = pred_length
        if self.pred_length != 1:
            raise ValueError("SimpleInformerP is trained as a one-step predictor; set pred_length=1.")
        self.input_dim = input_dim

        self.main_embedding = DataEmbedding(c_in=input_dim, d_model=d_model, dropout=dropout)
        self.main_encoder = nn.ModuleList(
            [
                SimplifiedEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.main_output = nn.Linear(d_model, input_dim)
        self.residual_path = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, self.seq_length, -1)

        encoded = self.main_embedding(x_flat)
        for layer in self.main_encoder:
            encoded, _ = layer(encoded)

        main_out = self.main_output(encoded[:, -1, :])
        residual_out = self.residual_path(x_flat[:, -1, :])
        return main_out + residual_out


class SimplifiedDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, padding=0, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, padding=0, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimplifiedUNet(nn.Module):
    def __init__(self, input_channels: int = 1, features: Sequence[int] = (16, 32, 64)) -> None:
        super().__init__()
        if len(features) < 1:
            raise ValueError("features must contain at least one channel size.")
        features = list(features)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for index, out_channels in enumerate(features):
            in_channels = input_channels if index == 0 else features[index - 1]
            self.encoders.append(SimplifiedDoubleConv(in_channels, out_channels))
            if index < len(features) - 1:
                self.pools.append(nn.MaxPool2d(2))

        self.bottleneck = SimplifiedDoubleConv(features[-1], features[-1])

        for index in range(len(features) - 1, 0, -1):
            self.up_convs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(features[index], features[index - 1], 1),
                )
            )
            self.decoders.append(SimplifiedDoubleConv(features[index - 1] * 2, features[index - 1]))

        self.final_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(features[0], 16, 3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, input_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for encoder, pool in zip(self.encoders[:-1], self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.encoders[-1](x)
        skip_connections.append(x)
        x = self.bottleneck(x)

        skip_connections = skip_connections[:-1][::-1]
        for index, (up_conv, decoder) in enumerate(zip(self.up_convs, self.decoders)):
            x = up_conv(x)
            skip = skip_connections[index]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.final_conv(x)


class PixelTemporalInformer(nn.Module):
    def __init__(
        self,
        seq_length: int,
        pred_length: int = 1,
        num_channels: int = 1,
        d_model: int = 32,
        n_heads: int = 2,
        e_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.pred_length = pred_length
        if self.pred_length != 1:
            raise ValueError("PixelTemporalInformer is trained as a one-step predictor; set pred_length=1.")

        self.embedding = DataEmbedding(c_in=num_channels, d_model=d_model, dropout=dropout)
        self.encoder = nn.ModuleList(
            [
                SimplifiedEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.output = nn.Linear(d_model, num_channels)
        self.residual = nn.Linear(num_channels, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.dim() != 5:
            raise ValueError("Expected input shape [B, T, C, H, W] or [B, T, H, W].")

        batch_size, seq_len, channels, height, width = x.shape
        if seq_len != self.seq_length:
            raise ValueError(f"Expected seq_length={self.seq_length}, got {seq_len}.")
        if channels != self.num_channels:
            raise ValueError(f"Expected num_channels={self.num_channels}, got {channels}.")

        sequence = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, seq_len, channels)
        encoded = self.embedding(sequence)
        for layer in self.encoder:
            encoded, _ = layer(encoded)

        output = self.output(encoded[:, -1]) + self.residual(sequence[:, -1])
        return output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


class WholeRasterForecastModel(nn.Module):
    """Forecast one full raster at a time without spatial patch slicing."""

    def __init__(
        self,
        seq_length: int = 10,
        pred_length: int = 1,
        num_channels: int = 1,
        d_model: int = 32,
        n_heads: int = 2,
        e_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.1,
        unet_features: Sequence[int] = (16, 32, 64),
        use_global_refiner: bool = False,
        global_refiner_features: Sequence[int] = (16, 32),
        residual_prediction: bool = False,
        **_legacy_patch_kwargs,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.pred_length = pred_length
        if self.pred_length != 1:
            raise ValueError("WholeRasterForecastModel is trained as a one-step predictor; set pred_length=1.")
        self.num_channels = num_channels
        self.residual_prediction = False

        self.temporal_model = PixelTemporalInformer(
            seq_length=seq_length,
            pred_length=pred_length,
            num_channels=num_channels,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.feature_reshape = nn.Linear(num_channels, num_channels)
        self.unet = SimplifiedUNet(input_channels=num_channels, features=unet_features)

    @classmethod
    def from_config(cls, config) -> "WholeRasterForecastModel":
        return cls(
            seq_length=config.seq_length,
            pred_length=config.pred_length,
            num_channels=config.num_channels,
            d_model=config.informer_d_model,
            n_heads=config.informer_n_heads,
            e_layers=config.informer_e_layers,
            d_ff=config.informer_d_ff,
            dropout=config.informer_dropout,
            unet_features=config.unet_features,
            use_global_refiner=config.use_global_refiner,
            global_refiner_features=config.global_refiner_features,
            residual_prediction=getattr(config, "residual_prediction", True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.dim() != 5:
            raise ValueError("Expected input shape [B, T, C, H, W] or [B, T, H, W].")

        batch_size, seq_len, channels, height, width = x.shape
        if seq_len != self.seq_length:
            raise ValueError(f"Expected seq_length={self.seq_length}, got {seq_len}.")
        if channels != self.num_channels:
            raise ValueError(f"Expected num_channels={self.num_channels}, got {channels}.")
        del batch_size
        informer_out = self.temporal_model(x)
        linear_feature = self.feature_reshape(
            informer_out.permute(0, 2, 3, 1).reshape(-1, channels)
        )
        spatial_feature = linear_feature.reshape(-1, height, width, channels).permute(0, 3, 1, 2).contiguous()
        final_out = self.unet(spatial_feature)
        return final_out.unsqueeze(1)


PatchwiseRasterForecastModel = WholeRasterForecastModel
SimpleInformerPUNetModel = WholeRasterForecastModel
