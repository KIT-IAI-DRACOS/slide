
import torch

from models.mamko.mamko import Mamba, weights_init

from ..basemodel import BaseModel
from ..static_feature_embedding import StaticFeatureEmbedding


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MamkoAdapter(BaseModel):
    def __init__(
            self, 

            pred_horizon, 
            old_horizon,
            latent_dim,
            state_dim,
            d_conv,
            future_num,
            stat_num,
            past_num,
            delta,
            hidden_dim,

            use_norm,
            use_exog_norm,
            use_static_exog,
            static_cardinalities,
            static_embedding_dims,
            lr,
            weight_decay,
            lr_scheduler,
            optimizer_step,
            optimizer_gamma,
            loss_function,
            shifted_u=False,
            **kwargs,
            ):
        super().__init__(
            loss_function=loss_function,
            **kwargs,
        )

        self.save_hyperparameters()

        act_dim = future_num
        state_dim += past_num
        self.past_num = past_num

        if stat_num > 0 and use_static_exog:
            self.static_embedding = StaticFeatureEmbedding(static_cardinalities, static_embedding_dims)
            act_dim += sum(static_embedding_dims)

        config = {
            'pred_horizon': pred_horizon,
            'old_horizon': old_horizon,
            'latent_dim': latent_dim,
            'state_dim': state_dim,
            'd_conv': d_conv,
            'act_dim': act_dim,
            'delta': int(delta * latent_dim),
            'hidden_dim': hidden_dim,
        }

        self.model = Mamba(config)
        self.model.apply(weights_init)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.opt_step = optimizer_step
        self.opt_gamma = optimizer_gamma
        self.use_static_exog = use_static_exog

        self.use_norm = use_norm
        self.use_exog_norm = use_exog_norm

        self.shifted_u = shifted_u

    def forward(self, batch):
        x, _y = batch

        state = x['past_target']
        u = x['future_features']

        if self.past_num > 0:
            state = torch.cat([state, x['past_features']], dim=-1)

        if self.use_norm:
            eps = 1e-5
            state_mean = state.mean(dim=1, keepdim=True).detach()
            state_std = torch.sqrt(torch.var(state, dim=1, keepdim=True, unbiased=False) + eps).detach()
            state = (state - state_mean) / state_std

        if self.use_exog_norm:
            eps = 1e-5
            u_mean = u.mean(dim=1, keepdim=True).detach()
            u_std = torch.sqrt(torch.var(u, dim=1, keepdim=True, unbiased=False) + eps).detach()
            u = (u - u_mean) / u_std

        if self.use_static_exog and hasattr(self, 'static_embedding'):
            stat_raw = x['static']

            stat_emb = self.static_embedding(stat_raw)

            stat_emb_expanded = stat_emb.unsqueeze(1).expand(-1, u.size(1), -1)

            u = torch.cat([u, stat_emb_expanded], dim=-1)

        if not self.shifted_u:
            pred = self.model(state, u)
        else:
            pred = self.model(state, u[:, 1:, :])

        if self.use_norm:
            pred = pred * state_std + state_mean
        
        if self.past_num > 0:
            pred = pred[:, :, : -self.past_num]

        return pred
        

    def configure_optimizers(self):
            
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler:
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.opt_step, gamma=self.opt_gamma) 

            return {"optimizer": opt, "lr_scheduler": sch}

        else:
            return opt
