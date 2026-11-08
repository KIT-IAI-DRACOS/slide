import torch

from ..basemodel import BaseModel
from ..static_feature_embedding import StaticFeatureEmbedding
from .gcgnet_model import GCGNetModel


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class GCGNetAdapter(BaseModel):
    def __init__(
            self, 
            endo_num,
            future_num,
            stat_num,
            patch_len,
            seq_len,
            pred_len,
            d_model,
            d_ff,
            rank,
            e_layers,
            n_heads,
            dropout,
            use_norm,
            use_exog_norm,
            use_future_exog,
            use_static_exog,
            static_cardinalities,
            static_embedding_dims,
            lr,
            loss_function,
            clamp_logvar=False,
            **kwargs,
            ):
        super().__init__(
            loss_function=loss_function,
            **kwargs,
        )

        self.save_hyperparameters()

        exog_num = future_num

        if stat_num > 0 and use_static_exog:
            self.static_embedding = StaticFeatureEmbedding(static_cardinalities, static_embedding_dims)
            exog_num += sum(static_embedding_dims)

        config = Config(
            enc_in=endo_num + exog_num,
            series_dim=endo_num,
            exog_num=exog_num,
            patch_len=patch_len,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            d_ff=d_ff,
            rank=rank,
            e_layers=e_layers,
            n_heads=n_heads,
            dropout=dropout,
            use_norm=use_norm,
            use_exog_norm=use_exog_norm,
            use_future_exog=use_future_exog,
            clamp_logvar=clamp_logvar,
        )

        self.model = GCGNetModel(config)
        self.seq_len = seq_len
        self.lr = lr
        self.use_static_exog = use_static_exog

    def forward(self, batch):
        x, y = batch

        endo_history = x['past_target']
        exog_history = x['future_features'][:,:self.seq_len,:]
        exog_future = x['future_features'][:,self.seq_len:,:]

        if self.use_static_exog and hasattr(self, 'static_embedding'):
            stat_raw = x['static']
            stat_emb = self.static_embedding(stat_raw)
            stat_emb_history_expanded = stat_emb.unsqueeze(1).expand(-1, endo_history.size(1), -1)
            stat_emb_future_expanded = stat_emb.unsqueeze(1).expand(-1, exog_future.size(1), -1)
            exog_history = torch.cat([exog_history, stat_emb_history_expanded], dim=-1)
            exog_future = torch.cat([exog_future, stat_emb_future_expanded], dim=-1)

        history = torch.cat([endo_history, exog_history], dim=-1)

        if self.training:
            return self.model(history, exog_future, y)
        else:
            return self.model(history, exog_future, y)[0]
        
    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one training step."""
        _x, y, *_ = batch
        y_hat, loss = self(batch)

        loss = self.loss_function(torch.squeeze(y), torch.squeeze(y_hat)) + loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=y.size(0),
        )
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        def lr_lambda(epoch: int):
            if epoch < 3:
                return 1.0
            return 0.9 ** (epoch - 3)

        sch = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda),
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_type3",
        }
        return {"optimizer": opt, "lr_scheduler": sch}
