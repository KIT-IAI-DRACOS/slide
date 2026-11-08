from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ..basemodel import BaseModel, LossName, ModelBatch
from ..static_feature_embedding import StaticFeatureEmbedding
from .timexer import timexer_model


class Config:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class TimexerAdapter(BaseModel):
    def __init__(
        self,
        endo_num: int,
        future_num: int,
        stat_num: int,
        patch_len: int,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_ff_factor: int,
        factor: int,
        activation: str,
        e_layers: int,
        n_heads: int,
        dropout: float,
        use_norm: bool,
        use_exog_norm: bool,
        use_future: bool,
        use_static_exog: bool,
        static_cardinalities: Sequence[int] | None,
        static_embedding_dims: Sequence[int] | None,
        lr: float,
        loss_function: LossName,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            loss_function=loss_function,
            **kwargs,
        )

        self.save_hyperparameters()

        exog_num = future_num

        if stat_num > 0 and use_static_exog:
            if static_cardinalities is None or static_embedding_dims is None:
                raise ValueError(
                    "Static cardinalities and embedding dimensions are required when static inputs are used."
                )
            self.static_embedding = StaticFeatureEmbedding(static_cardinalities, static_embedding_dims)
            exog_num += sum(static_embedding_dims)

        config = Config(
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            enc_in=endo_num + exog_num,
            series_dim=endo_num,
            freq=None,  # not used in model
            factor=factor,
            activation=activation,
            exog_num=exog_num,
            d_model=d_model,
            d_ff=d_ff_factor * d_model,
            e_layers=e_layers,
            n_heads=n_heads,
            dropout=dropout,
            use_norm=use_norm,
            use_exog_norm=use_exog_norm,
            use_future=use_future,
        )

        self.model = timexer_model(config)
        self.seq_len = seq_len
        self.lr = lr
        self.use_static_exog = use_static_exog

    def forward(self, batch: ModelBatch) -> torch.Tensor:
        x, _y = batch

        endo_history = x["past_target"]
        exog_history = x["future_features"][:, : self.seq_len, :]
        exog_future = x["future_features"][:, self.seq_len :, :]

        if self.use_static_exog and hasattr(self, "static_embedding"):
            stat_raw = x["static"]
            stat_emb = self.static_embedding(stat_raw)
            stat_emb_history_expanded = stat_emb.unsqueeze(1).expand(-1, endo_history.size(1), -1)
            stat_emb_future_expanded = stat_emb.unsqueeze(1).expand(-1, exog_future.size(1), -1)
            exog_history = torch.cat([exog_history, stat_emb_history_expanded], dim=-1)
            exog_future = torch.cat([exog_future, stat_emb_future_expanded], dim=-1)

        history = torch.cat([endo_history, exog_history], dim=-1)

        return self.model(history, exog_future)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Match the official TimeXer repository's "type1" schedule. GCGNet's bundled
        # TimeXer training wrapper uses its "type3" schedule instead.

        def lr_lambda(epoch: int) -> float:
            if epoch <= 0:
                return 1.0
            return 0.5 ** (epoch - 1)

        sch = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda),
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_adjust",
        }
        return {"optimizer": opt, "lr_scheduler": sch}
