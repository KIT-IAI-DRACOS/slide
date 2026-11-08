"""SLiDE with optional replacements for the dynamics, initial state, and exogenous map."""

from collections.abc import Sequence
from typing import Any

import torch
from einops import rearrange

from models.basemodel import LossName
from models.slide.blocks.mlp import MLP
from models.slide.blocks.rope_transformer import RoPETransformerEncoder
from models.slide.slide import KoopBlock, SchedulerName
from models.slide.slide import Model as SlideModel


class KoopBlockAblation(KoopBlock):
    """Koopman block with the linear path plus ablation branches."""

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
        non_linear: bool = False,
        mlp_initial_state: bool = False,
        transformer_initial_state: bool = False,
        transformer_n_layers: int = 2,
        transformer_n_heads: int = 8,
        state_dependent_ex_encode: bool = False,
    ) -> None:
        if mlp_initial_state and transformer_initial_state:
            raise ValueError(
                "mlp_initial_state and transformer_initial_state cannot be enabled together."
            )
        if state_dependent_ex_encode and ex_in <= 0:
            raise ValueError(
                "state_dependent_ex_encode=True requires exogenous inputs."
            )

        super().__init__(
            num_blocks,
            enc_in=enc_in,
            ex_in=ex_in,
            dec_out=dec_out,
            input_len=input_len,
            pred_len=pred_len,
            seg_len=seg_len,
            dynamic_dim=dynamic_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            param_dim=param_dim,
        )
        self.dynamic_dim = dynamic_dim
        self.non_linear = non_linear
        self.mlp_initial_state = mlp_initial_state
        self.transformer_initial_state = transformer_initial_state
        self.state_dependent_ex_encode = state_dependent_ex_encode

        if non_linear:
            self.Wh = MLP(
                f_in=dynamic_dim,
                f_out=dynamic_dim,
                hidden_dim=dynamic_dim * 2,
                hidden_layers=hidden_layers,
                dropout=dropout,
            )
            self.Wz = self.Wh

        if state_dependent_ex_encode:
            self.encoder_u = None
            self.u_patch_dim = seg_len * ex_in
            self.encoder_u_state_dep = MLP(
                f_in=self.u_patch_dim + dynamic_dim,
                f_out=dynamic_dim * self.u_patch_dim,
                hidden_dim=dynamic_dim * 2,
                hidden_layers=hidden_layers,
                dropout=dropout,
            )

        if mlp_initial_state:
            raw_len_past = self.seg_len * self.freq
            mlp_h_in_dim = raw_len_past * enc_in
            if ex_in > 0:
                mlp_h_in_dim += raw_len_past * ex_in
            self.h_mlp = MLP(
                f_in=mlp_h_in_dim,
                f_out=dynamic_dim,
                hidden_dim=dynamic_dim * 2,
                hidden_layers=hidden_layers,
                dropout=dropout,
            )

        if transformer_initial_state:
            self.h_transformer = RoPETransformerEncoder(
                d_model=dynamic_dim,
                n_layers=transformer_n_layers,
                n_heads=transformer_n_heads,
                dropout=dropout,
            )

    def _state_dependent_force(self, u_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Exogenous forcing p = B(u, state) u, with B scaled by 1/sqrt(u_patch_dim)."""
        b_t = self.encoder_u_state_dep(torch.cat([u_t, state], dim=-1))
        b_t = b_t.view(u_t.shape[0], self.dynamic_dim, self.u_patch_dim)
        b_t = b_t / (self.u_patch_dim ** 0.5)
        return torch.bmm(b_t, u_t.unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        ex: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            not self.mlp_initial_state
            and not self.transformer_initial_state
            and not self.state_dependent_ex_encode
        ):
            return super().forward(x, ex)

        _B, L, _C = x.shape
        ex_raw = ex

        res = torch.cat((x[:, L - self.padding_len :, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)
        res = rearrange(torch.stack(res, dim=1), "b f p c -> b f (p c)")
        d = self.encoder_y(res)

        ex_segs = None
        if ex is not None and self.encoder_u is not None and not self.state_dependent_ex_encode:
            ex_encoded = ex.chunk(self.freq + self.step, dim=1)
            ex_encoded = rearrange(torch.stack(ex_encoded, dim=1), "b f p c -> b f (p c)")
            p = self.encoder_u(ex_encoded)
        elif ex is not None and self.state_dependent_ex_encode:
            ex_chunked = ex.chunk(self.freq + self.step, dim=1)
            ex_segs = rearrange(torch.stack(ex_chunked, dim=1), "b f p c -> b f (p c)")
            p = None
        else:
            p = None

        if self.transformer_initial_state:
            tokens = d[:, : self.freq, :]
            if p is not None:
                tokens = tokens + p[:, : self.freq, :]
            h = self.h_transformer(tokens)[:, -1, :]
        elif self.mlp_initial_state:
            res_raw = torch.cat((x[:, L - self.padding_len :, :], x), dim=1)
            x_raw_flat = rearrange(res_raw, "b l c -> b (l c)")
            if ex_raw is not None and self.encoder_u is not None:
                ex_past_raw = ex_raw[:, : self.seg_len * self.freq, :]
                ex_raw_flat = rearrange(ex_past_raw, "b l c -> b (l c)")
                h_mlp_inp = torch.cat([x_raw_flat, ex_raw_flat], dim=-1)
            else:
                h_mlp_inp = x_raw_flat
            h = self.h_mlp(h_mlp_inp)
        else:
            h = torch.zeros(d.shape[0], d.shape[2], device=d.device)
            for t in range(self.freq):
                h_prime = self.Wh(h) + d[:, t, :]
                if p is not None:
                    h_prime = h_prime + p[:, t, :]
                elif ex_segs is not None and self.state_dependent_ex_encode:
                    h_prime = h_prime + self._state_dependent_force(ex_segs[:, t, :], h)
                h = h_prime

        z_outputs = []
        z = h
        for t in range(self.step):
            z_prime = self.Wz(z)
            if p is not None:
                z_prime = z_prime + p[:, self.freq + t, :]
            elif ex_segs is not None and self.state_dependent_ex_encode:
                z_prime = z_prime + self._state_dependent_force(ex_segs[:, self.freq + t, :], z)
            z = z_prime
            z_outputs.append(z.unsqueeze(1))

        z_pred = torch.cat(z_outputs, dim=1)
        x_pred = self.decoder(z_pred)
        x_pred = rearrange(
            x_pred,
            "b s (p c q) -> b (s p) c q",
            c=self.dec_out,
            q=self.param_dim,
        )
        if self.param_dim == 1:
            x_pred = x_pred.squeeze(-1)
        return x_pred


class Model(SlideModel):
    """SLiDE model with optional ablation switches. Defaults match the linear model."""

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
        non_linear: bool = False,
        mlp_initial_state: bool = False,
        transformer_initial_state: bool = False,
        transformer_n_layers: int = 2,
        transformer_n_heads: int = 8,
        state_dependent_ex_encode: bool = False,
        **kwargs: Any,
    ) -> None:
        if mlp_initial_state and transformer_initial_state:
            raise ValueError(
                "mlp_initial_state and transformer_initial_state cannot be enabled together."
            )
        super().__init__(
            num_targets=num_targets,
            seq_len=seq_len,
            pred_len=pred_len,
            seg_len=seg_len,
            num_past_features=num_past_features,
            num_future_ex=num_future_ex,
            num_static_ex=num_static_ex,
            static_cardinalities=static_cardinalities,
            static_embedding_dims=static_embedding_dims,
            num_blocks=num_blocks,
            dynamic_dim=dynamic_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            lr=lr,
            lr_scheduler=lr_scheduler,
            adamw_wd=adamw_wd,
            loss_function=loss_function,
            stationarize=stationarize,
            stationarize_ex=stationarize_ex,
            **kwargs,
        )
        self.non_linear = non_linear
        self.mlp_initial_state = mlp_initial_state
        self.transformer_initial_state = transformer_initial_state
        self.transformer_n_layers = transformer_n_layers
        self.transformer_n_heads = transformer_n_heads
        self.state_dependent_ex_encode = state_dependent_ex_encode

        num_ex = num_future_ex
        if num_static_ex > 0:
            num_ex += sum(static_embedding_dims)

        self.blocks = torch.nn.ModuleList(
            [
                KoopBlockAblation(
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
                    non_linear=self.non_linear,
                    mlp_initial_state=self.mlp_initial_state,
                    transformer_initial_state=self.transformer_initial_state,
                    transformer_n_layers=self.transformer_n_layers,
                    transformer_n_heads=self.transformer_n_heads,
                    state_dependent_ex_encode=self.state_dependent_ex_encode,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.save_hyperparameters()
