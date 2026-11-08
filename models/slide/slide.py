import math
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from models.slide.blocks.mlp import MLP

from ..basemodel import BaseModel, LossName, ModelBatch
from ..static_feature_embedding import StaticFeatureEmbedding

SchedulerName: TypeAlias = Literal["no_scheduler", "exponential"]

class KoopBlock(nn.Module):
    """
    Koopman Block
    """
    
    def __init__(
        self,
        num_blocks: int,
        enc_in: int = 8,
        ex_in: int = 3,
        dec_out: int = 8,
        input_len: int = 96,
        pred_len: int = 96,
        seg_len: int = 16,
        dynamic_dim: int = 128,
        hidden_layers: int = 1,
        dropout: float = 0.05,
        param_dim: int = 1,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.param_dim = param_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_out = dec_out
        self.seg_len = seg_len
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)  # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len

        # Encoder: projects directly to dynamic_dim (latent dimension)
        # Channel-mixed mode
        self.encoder_y = MLP(
            f_in=seg_len * enc_in,
            f_out=dynamic_dim,
            hidden_dim=dynamic_dim*2,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )

        # Koopman linear transformation matrix Wh (for hidden state update)
        # h[k'] = Wh @ h[k'-1] + d[k'] + p[k']
        self.Wh = nn.Linear(dynamic_dim, dynamic_dim, bias=False)  # This is K
        self.Wz = self.Wh # same as Wh

        # Exogenous input encoder (if provided)
        if ex_in > 0:
            self.encoder_u = MLP(
                f_in=seg_len * ex_in,
                f_out=dynamic_dim,
                hidden_dim=dynamic_dim*2,
                hidden_layers=hidden_layers,
                dropout=dropout,
            )
        else:
            self.encoder_u = None

        # Decoder: latent state to output
        self.decoder = MLP(
            f_in=dynamic_dim,
            f_out=seg_len * dec_out * self.param_dim,
            hidden_dim=dynamic_dim*2,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        ex: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode the history and exogenous inputs, then forecast one horizon.
        """
        # x: B L C = [32, 96, 7]
        # ex: B (L+T) C = [32, 192, 3]
        _B, L, _C = x.shape

        # Padding to make length divisible by seg_len
        res = torch.cat((x[:, L - self.padding_len :, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)  # F x B P C, P means seg_len

        res = rearrange(torch.stack(res, dim=1), "b f p c -> b f (p c)")  # B F PC

        # Encode y values to latent dimension
        d = self.encoder_y(res)  # B F H

        # Encode exogenous if provided
        if ex is not None and self.encoder_u is not None:
            ex = ex.chunk(self.freq + self.step, dim=1)  # (F+S) x B P C
            ex = rearrange(torch.stack(ex, dim=1), "b f p c -> b f (p c)")  # B F (P C)
            p = self.encoder_u(ex)  # B (F+S) H
        else:
            p = None

        # Compute hidden state h_k by iterating through time
        # h[k'] = W @ h[k'-1] + d[k'] + p[k']
        h = torch.zeros(d.shape[0], d.shape[2], device=d.device)  # B H

        for t in range(self.freq):
            h_prime = self.Wh(h) + d[:, t, :]
            if p is not None:
                h_prime = h_prime + p[:, t, :]
            h = h_prime

        # Now forecast: z[k,h] = W @ z[k,h-1] + p[k,h]
        # Initialize z[k,0] = h
        z_outputs = []
        z = h
        for t in range(self.step):
            z_prime = self.Wz(z)
            if p is not None:
                z_prime = z_prime + p[:, self.freq + t, :]
            z = z_prime
            z_outputs.append(z.unsqueeze(1))
        
        z_pred = torch.cat(z_outputs, dim=1)  # B S H
        
        # Decode
        x_pred = self.decoder(z_pred)  # B S (P*C*param_dim)

        # Rearrange to output shape
        x_pred = rearrange(
            x_pred,
            "b s (p c q) -> b (s p) c q",
            c=self.dec_out,
            q=self.param_dim,
        )

        if self.param_dim == 1:
            x_pred = x_pred.squeeze(-1)

        return x_pred


class Model(BaseModel):
    """
    SLiDE
    """
    def __init__(
        self,
        num_targets: int,
        seq_len: int,
        pred_len: int,
        seg_len: int,
        num_past_features: int,
        num_future_ex: int,
        num_static_ex: int,
        static_cardinalities: Sequence[int] | None,
        static_embedding_dims: Sequence[int] | None,
        num_blocks: int,
        dynamic_dim: int,
        hidden_layers: int,
        dropout: float,
        lr: float,
        lr_scheduler: SchedulerName,
        adamw_wd: float,
        loss_function: LossName,
        stationarize: bool = True,
        stationarize_ex: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            loss_function=loss_function,
            **kwargs,
        )
        self.save_hyperparameters()
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.adamw_wd = adamw_wd
        self.param_dim = self.loss_function.outputsize
        self.enc_in = num_targets + num_past_features
        self.dec_out = num_targets
        self.input_len = seq_len
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.num_blocks = num_blocks
        self.dynamic_dim = dynamic_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.stationarize = stationarize
        self.stationarize_ex = stationarize_ex

        num_ex = num_future_ex
        if num_static_ex > 0:
            self.static_embedding = StaticFeatureEmbedding(static_cardinalities, static_embedding_dims)
            num_ex += sum(static_embedding_dims)

        # initialize KoopBlock blocks
        self.blocks = nn.ModuleList(
            [
                KoopBlock(
                    self.num_blocks,
                    enc_in=self.enc_in,
                    ex_in=num_ex,
                    dec_out=self.dec_out,
                    input_len=self.input_len,
                    pred_len=self.pred_len,
                    seg_len=self.seg_len,
                    dynamic_dim=self.dynamic_dim,
                    hidden_layers=self.hidden_layers,
                    dropout=self.dropout,
                    param_dim=self.param_dim,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, batch: ModelBatch) -> torch.Tensor:
        x, _y = batch

        x_enc = x["past_target"]  # [batch_size, seq_len, C]
        past_feat = x.get("past_features", None)  # [batch_size, seq_len, num_past_ex]
        future_feat = x.get("future_features", None) # [batch_size, seq_len + pred_len, num_future_ex]
        stat = x.get("static", None) # [B, num_static_ex]

        # series stationarization
        if self.stationarize:
            x_enc, mean_enc, std_enc = self._stationarize(x_enc)  # [B, L, C], [B, 1, C], [B, 1, C]

        # past features
        if self._is_nonempty(past_feat):
            x_enc = torch.cat([x_enc, past_feat], dim=-1)  # [batch_size, seq_len, C + num_past_ex]

        ex = None
        # future features
        if self._is_nonempty(future_feat):
            ex = future_feat
            if self.stationarize_ex:
                ex, _mean_ex, _std_ex = self._stationarize(ex)  # [B, L+H, C + num_future_ex]

        # static features
        if self._is_nonempty(stat):
            stat_emb = self.static_embedding(stat)  # [B, E]
            LH = self.input_len + self.pred_len
            stat_emb_ex = stat_emb.unsqueeze(1).expand(-1, LH, -1)  # [B, L+H, E]

            if ex is None:
                ex = stat_emb_ex
            else:
                ex = torch.cat([ex, stat_emb_ex], dim=-1)

        encoder_inputs = [x_enc] * self.num_blocks

        # Process the input through each block
        results = [self.blocks[i](encoder_inputs[i], ex) for i in range(self.num_blocks)]

        # Sum results from all blocks
        combined_x_pred = sum(results)  # [B, H, C] or [B, H, C, param_dim]

        # Series Stationarization adopted from NSformer
        if self.stationarize: 
            pred = combined_x_pred * std_enc[:, :, : self.dec_out] + mean_enc[:, :, : self.dec_out]
        else:
            pred = combined_x_pred

        return pred

    ### helper functions ###
    @staticmethod
    def _is_nonempty(t: torch.Tensor | None) -> bool:
        return t is not None and t.numel() > 0

    @staticmethod
    def _stationarize(z: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [B, L, C] with C>0
        mean = z.mean(dim=1, keepdim=True).detach()
        zc = z - mean
        std = torch.sqrt(torch.var(zc, dim=1, keepdim=True, unbiased=False) + eps).detach()
        zn = zc / std
        return zn, mean, std

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.adamw_wd)
        if self.lr_scheduler == "no_scheduler":
            return optimizer
        elif self.lr_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
