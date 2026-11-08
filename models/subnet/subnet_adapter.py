
import torch

from models.subnet.subnet import SUBNET, SUBNET_CT, Koopman_SUBNET

from ..basemodel import BaseModel
from ..static_feature_embedding import StaticFeatureEmbedding


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class SUBNETAdapter(BaseModel):
    def __init__(
            self, 
            model_name,
            pred_horizon, 
            input_length,
            latent_dim,
            state_dim,
            future_num,
            stat_num,
            past_num,
            feedthrough,

            use_norm,
            use_exog_norm,
            use_static_exog,
            static_cardinalities,
            static_embedding_dims,
            lr,
            loss_function,
            b_depends_on_u=False,
            **kwargs,
            ):
        super().__init__(
            loss_function=loss_function,
            **kwargs,
        )

        self.save_hyperparameters()

        self.model_name = model_name

        act_dim = future_num
        state_dim += past_num
        self.past_num = past_num

        if stat_num > 0 and use_static_exog:
            self.static_embedding = StaticFeatureEmbedding(static_cardinalities, static_embedding_dims)
            act_dim += sum(static_embedding_dims)

        if model_name == "ctsubnet":
            self.model = SUBNET_CT(nu=act_dim, ny=state_dim, nx=latent_dim, nb=input_length, na=input_length, feedthrough=feedthrough)
        elif model_name == "subnet":
            self.model = SUBNET(nu=act_dim, ny=state_dim, nx=latent_dim, nb=input_length, na=input_length, feedthrough=feedthrough)
        elif model_name == "koopsubnet":
            self.model = Koopman_SUBNET(nu=act_dim, ny=state_dim, nx=latent_dim, nb=input_length, na=input_length, B_depends_on_u=b_depends_on_u, feedthrough=feedthrough)

        self.lr = lr

        self.use_static_exog = use_static_exog

        self.use_norm = use_norm
        self.use_exog_norm = use_exog_norm

        self.input_length = input_length
        self.pred_horizon = pred_horizon

        

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

        u_past = u[:,:self.input_length,:]
        u_future = u[:,self.input_length:,:]

        if self.model_name == "ctsubnet":
            pred = self.model(upast=u_past, ypast=state, ufuture=u_future, sampling_time=1)
        elif self.model_name == "subnet" or self.model_name == "koopsubnet":
            pred = self.model(upast=u_past, ypast=state, ufuture=u_future)

        if self.use_norm:
            pred = pred * state_std + state_mean
        
        if self.past_num > 0:
            pred = pred[:, :, : -self.past_num]

        return pred
        

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt
