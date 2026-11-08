# SPDX-FileCopyrightText: Copyright 2024 Nixtla
# SPDX-License-Identifier: Apache-2.0
#
# This file is based on code from Neuralforecast:
# https://github.com/Nixtla/neuralforecast
#
# Modified from the original.

__all__ = ['TemporalMixing', 'FeatureMixing', 'MixingLayer', 'MixingLayerWithStaticExogenous', 'TSMixerx']


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

### Adjusted import paths to match project structure
# from ..common._base_model import BaseModel
# from ..common._modules import RevINMultivariate
# from ..losses.pytorch import MAE
from ..basemodel import BaseModel
from ..static_feature_embedding import StaticFeatureEmbedding
###

class TemporalMixing(nn.Module):
    """
    TemporalMixing
    """

    def __init__(self, num_features, h, dropout):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(normalized_shape=(h, num_features))
        self.temporal_lin = nn.Linear(h, h)
        self.temporal_drop = nn.Dropout(dropout)

    def forward(self, input):
        x = input.permute(0, 2, 1)  # [B, h, C] -> [B, C, h]
        x = F.relu(self.temporal_lin(x))  # [B, C, h] -> [B, C, h]
        x = x.permute(0, 2, 1)  # [B, C, h] -> [B, h, C]
        x = self.temporal_drop(x)  # [B, h, C] -> [B, h, C]

        return self.temporal_norm(x + input)


class FeatureMixing(nn.Module):
    """
    FeatureMixing
    """

    def __init__(self, in_features, out_features, h, dropout, ff_dim):
        super().__init__()
        self.feature_lin_1 = nn.Linear(in_features=in_features, out_features=ff_dim)
        self.feature_lin_2 = nn.Linear(in_features=ff_dim, out_features=out_features)
        self.feature_drop_1 = nn.Dropout(p=dropout)
        self.feature_drop_2 = nn.Dropout(p=dropout)
        self.linear_project_residual = False
        if in_features != out_features:
            self.project_residual = nn.Linear(
                in_features=in_features, out_features=out_features
            )
            self.linear_project_residual = True

        self.feature_norm = nn.LayerNorm(normalized_shape=(h, out_features))

    def forward(self, input):
        x = F.relu(self.feature_lin_1(input))  # [B, h, C_in] -> [B, h, ff_dim]
        x = self.feature_drop_1(x)  # [B, h, ff_dim] -> [B, h, ff_dim]
        x = self.feature_lin_2(x)  # [B, h, ff_dim] -> [B, h, C_out]
        x = self.feature_drop_2(x)  # [B, h, C_out] -> [B, h, C_out]
        if self.linear_project_residual:
            input = self.project_residual(input)  # [B, h, C_in] -> [B, h, C_out]

        return self.feature_norm(x + input)


class MixingLayer(nn.Module):
    """
    MixingLayer
    """

    def __init__(self, in_features, out_features, h, dropout, ff_dim):
        super().__init__()
        # Mixing layer consists of a temporal and feature mixer
        self.temporal_mixer = TemporalMixing(
            num_features=in_features, h=h, dropout=dropout
        )
        self.feature_mixer = FeatureMixing(
            in_features=in_features,
            out_features=out_features,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )

    def forward(self, input):
        x = self.temporal_mixer(input)  # [B, h, C_in] -> [B, h, C_in]
        x = self.feature_mixer(x)  # [B, h, C_in] -> [B, h, C_out]
        return x


class MixingLayerWithStaticExogenous(nn.Module):
    """
    MixingLayerWithStaticExogenous
    """

    def __init__(self, h, dropout, ff_dim, stat_input_size):
        super().__init__()
        # Feature mixer for the static exogenous variables
        self.feature_mixer_stat = FeatureMixing(
            in_features=stat_input_size,
            out_features=ff_dim,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )
        # Mixing layer consists of a temporal and feature mixer
        self.temporal_mixer = TemporalMixing(
            num_features=2 * ff_dim, h=h, dropout=dropout
        )
        self.feature_mixer = FeatureMixing(
            in_features=2 * ff_dim,
            out_features=ff_dim,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )

    def forward(self, inputs):
        input, stat_exog = inputs
        x_stat = self.feature_mixer_stat(stat_exog)  # [B, h, S] -> [B, h, ff_dim]
        x = torch.cat(
            (input, x_stat), dim=2
        )  # [B, h, ff_dim] + [B, h, ff_dim] -> [B, h, 2 * ff_dim]
        x = self.temporal_mixer(x)  # [B, h, 2 * ff_dim] -> [B, h, 2 * ff_dim]
        x = self.feature_mixer(x)  # [B, h, 2 * ff_dim] -> [B, h, ff_dim]
        return (x, stat_exog)


class ReversibleInstanceNorm1d(nn.Module):
    def __init__(self, n_series, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, 1, 1, n_series)))
        self.bias = nn.Parameter(torch.zeros((1, 1, 1, n_series)))
        self.eps = eps

    def forward(self, x):
        # Batch statistics
        self.batch_mean = torch.mean(x, axis=2, keepdim=True).detach()
        self.batch_std = torch.sqrt(
            torch.var(x, axis=2, keepdim=True, unbiased=False) + self.eps
        ).detach()

        # Instance normalization
        x = x - self.batch_mean
        x = x / self.batch_std
        x = x * self.weight
        x = x + self.bias

        return x

    def reverse(self, x):
        # Reverse the normalization
        x = x - self.bias
        x = x / self.weight
        x = x * self.batch_std
        x = x + self.batch_mean

        return x


class TSMixerx(BaseModel):
    """TSMixerx

    Time-Series Mixer exogenous (`TSMixerx`) is a MLP-based multivariate time-series forecasting model, with capability for additional exogenous inputs. `TSMixerx` jointly learns temporal and cross-sectional representations of the time-series by repeatedly combining time- and feature information using stacked mixing layers. A mixing layer consists of a sequential time- and feature Multi Layer Perceptron (`MLP`).

    Args:
        h (int): forecast horizon.
        input_size (int): considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].
        n_series (int): number of time-series.
        futr_exog_list (str list): future exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        stat_exog_list (str list): static exogenous columns.
        exclude_insample_y (bool): if True excludes insample_y from the model.
        n_block (int): number of mixing layers in the model.
        ff_dim (int): number of units for the second feed-forward layer in the feature MLP.
        dropout (float): dropout rate between (0, 1) .
        revin (bool): if True uses Reverse Instance Normalization on `insample_y` and applies it to the outputs.
        loss (PyTorch module): instantiated train loss class from [losses collection](./losses.pytorch).
        valid_loss (PyTorch module): instantiated valid loss class from [losses collection](./losses.pytorch).
        max_steps (int): maximum number of training steps.
        learning_rate (float): Learning rate between (0, 1).
        num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
        val_check_steps (int): Number of training steps between every validation loss check.
        batch_size (int): number of different series in each batch.
        valid_batch_size (int): number of different series in each validation and test batch, if None uses batch_size.
        windows_batch_size (int): number of windows to sample in each training batch.
        inference_windows_batch_size (int): number of windows to sample in each inference batch, -1 uses all.
        start_padding_enabled (bool): if True, the model will pad the time series with zeros at the beginning, by input size.
        training_data_availability_threshold (Union[float, List[float]]): minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).
        step_size (int): step size between each window of temporal data.
        scaler_type (str): type of scaler for temporal inputs normalization see [temporal scalers](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_scalers.py).
        random_seed (int): random_seed for pytorch initializer and numpy generators.
        drop_last_loader (bool): if True `TimeSeriesDataLoader` drops last non-full batch.
        alias (str): optional,  Custom name of the model.
        optimizer (Subclass of 'torch.optim.Optimizer'): optional, user specified optimizer instead of the default choice (Adam).
        optimizer_kwargs (dict): optional, list of parameters used by the user specified `optimizer`.
        lr_scheduler (Subclass of 'torch.optim.lr_scheduler.LRScheduler'): optional, user specified lr_scheduler instead of the default choice (StepLR).
        lr_scheduler_kwargs (dict): optional, list of parameters used by the user specified `lr_scheduler`.
        dataloader_kwargs (dict): optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`.
        **trainer_kwargs (int):  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

    References:
        - [Chen, Si-An, Chun-Liang Li, Nate Yoder, Sercan O. Arik, and Tomas Pfister (2023). "TSMixer: An All-MLP Architecture for Time Series Forecasting."](http://arxiv.org/abs/2303.06053)

    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = True  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    # def __init__(
    #     self,
    #     h,
    #     input_size,
    #     n_series,
    #     futr_exog_list=None,
    #     hist_exog_list=None,
    #     stat_exog_list=None,
    #     exclude_insample_y=False,
    #     n_block=2,
    #     ff_dim=64,
    #     dropout=0.0,
    #     revin=True,
    #     loss=MAE(),
    #     valid_loss=None,
    #     max_steps: int = 1000,
    #     learning_rate: float = 1e-3,
    #     num_lr_decays: int = -1,
    #     early_stop_patience_steps: int = -1,
    #     val_check_steps: int = 100,
    #     batch_size: int = 32,
    #     valid_batch_size: Optional[int] = None,
    #     windows_batch_size=32,
    #     inference_windows_batch_size=32,
    #     start_padding_enabled=False,
    #     training_data_availability_threshold=0.0,
    #     step_size: int = 1,
    #     scaler_type: str = "identity",
    #     random_seed: int = 1,
    #     drop_last_loader: bool = False,
    #     alias: Optional[str] = None,
    #     optimizer=None,
    #     optimizer_kwargs=None,
    #     lr_scheduler=None,
    #     lr_scheduler_kwargs=None,
    #     dataloader_kwargs=None,
    #     **trainer_kwargs
    # ):

    #     # Inherit BaseMultvariate class
    #     super(TSMixerx, self).__init__(
    #         h=h,
    #         input_size=input_size,
    #         n_series=n_series,
    #         futr_exog_list=futr_exog_list,
    #         hist_exog_list=hist_exog_list,
    #         stat_exog_list=stat_exog_list,
    #         exclude_insample_y=exclude_insample_y,
    #         loss=loss,
    #         valid_loss=valid_loss,
    #         max_steps=max_steps,
    #         learning_rate=learning_rate,
    #         num_lr_decays=num_lr_decays,
    #         early_stop_patience_steps=early_stop_patience_steps,
    #         val_check_steps=val_check_steps,
    #         batch_size=batch_size,
    #         valid_batch_size=valid_batch_size,
    #         windows_batch_size=windows_batch_size,
    #         inference_windows_batch_size=inference_windows_batch_size,
    #         start_padding_enabled=start_padding_enabled,
    #         training_data_availability_threshold=training_data_availability_threshold,
    #         step_size=step_size,
    #         scaler_type=scaler_type,
    #         random_seed=random_seed,
    #         drop_last_loader=drop_last_loader,
    #         alias=alias,
    #         optimizer=optimizer,
    #         optimizer_kwargs=optimizer_kwargs,
    #         lr_scheduler=lr_scheduler,
    #         lr_scheduler_kwargs=lr_scheduler_kwargs,
    #         dataloader_kwargs=dataloader_kwargs,
    #         **trainer_kwargs
    #     )
        ### Adjusted init to match project structure ###
    def __init__(
            self,
            h,
            input_size,
            n_series,
            revin, 
            revin_exo,
            ff_dim,
            dropout,
            n_block,
            futr_exog_size,
            hist_exog_size,
            stat_exog_size,
            static_cardinalities,
            static_embedding_dims,
            lr,
            loss_function,
            **kwargs,
    ):
        super(TSMixerx, self).__init__(
            loss_function=loss_function,
            **kwargs,
        )
        self.save_hyperparameters()
        self.lr = lr
        self.futr_exog_size = futr_exog_size
        self.hist_exog_size = hist_exog_size
        self.stat_exog_size = 0
        self.loss = self.loss_function
        self.loss.outputsize_multiplier = self.loss_function.outputsize

        # add static feature embedding module
        if stat_exog_size > 0:
            self.static_embedding = StaticFeatureEmbedding(static_cardinalities, static_embedding_dims) 
            self.stat_exog_size = sum(static_embedding_dims)  
        else:
            self.stat_exog_size = 0
        #################################################


        # Reversible InstanceNormalization layer for target
        self.revin = revin
        if self.revin:
            self.norm = RevINMultivariate(num_features=n_series, affine=True)
        
        # Reversible InstanceNormalization layer for exogenous inputs
        self.revin_exo = revin_exo
        if self.revin_exo:
            if self.hist_exog_size > 0:
                self.norm_hist_exog = RevINMultivariate(num_features=self.hist_exog_size, affine=True)
            if self.futr_exog_size > 0:
                self.norm_futr_exog = RevINMultivariate(num_features=self.futr_exog_size, affine=True)

        # Forecast horizon
        self.h = h

        # Temporal projection and feature mixing of historical variables
        self.temporal_projection = nn.Linear(in_features=input_size, out_features=h)

        self.feature_mixer_hist = FeatureMixing(
            ### adjust in_features
            # in_features=n_series * (1 + self.hist_exog_size + self.futr_exog_size),
            in_features=n_series + self.hist_exog_size + self.futr_exog_size,
            ###
            out_features=ff_dim,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )
        first_mixing_ff_dim_multiplier = 1

        # Feature mixing of future variables
        if self.futr_exog_size > 0:
            self.feature_mixer_futr = FeatureMixing(
                ### adjust in_features
                # in_features=n_series * self.futr_exog_size,
                in_features=self.futr_exog_size,
                ###
                out_features=ff_dim,
                h=h,
                dropout=dropout,
                ff_dim=ff_dim,
            )
            first_mixing_ff_dim_multiplier += 1

        # Feature mixing of static variables
        if self.stat_exog_size > 0:
            self.feature_mixer_stat = FeatureMixing(
                ### adjust in_features to match static feature input
                # in_features=self.stat_exog_size * n_series,
                in_features=self.stat_exog_size,
                out_features=ff_dim,
                h=h,
                dropout=dropout,
                ff_dim=ff_dim,
            )
            first_mixing_ff_dim_multiplier += 1

        # First mixing layer
        self.first_mixing = MixingLayer(
            in_features=first_mixing_ff_dim_multiplier * ff_dim,
            out_features=ff_dim,
            h=h,
            dropout=dropout,
            ff_dim=ff_dim,
        )

        # Mixing layer block
        if self.stat_exog_size > 0:
            mixing_layers = [
                MixingLayerWithStaticExogenous(
                    h=h,
                    dropout=dropout,
                    ff_dim=ff_dim,
                    ### adjust in_features to match static feature input
                    # stat_input_size=self.stat_exog_size * n_series,
                    stat_input_size=self.stat_exog_size,
                )
                for _ in range(n_block)
            ]
        else:
            mixing_layers = [
                MixingLayer(
                    in_features=ff_dim,
                    out_features=ff_dim,
                    h=h,
                    dropout=dropout,
                    ff_dim=ff_dim,
                )
                for _ in range(n_block)
            ]

        self.mixing_block = nn.Sequential(*mixing_layers)

        # Linear output with Loss dependent dimensions
        self.out = nn.Linear(
            in_features=ff_dim, out_features=self.loss.outputsize_multiplier * n_series
        )

    def forward(self, windows_batch):
        # Parse batch
        # x = windows_batch[
        #     "insample_y"
        # ]  #   [batch_size (B), input_size (L), n_series (N)]
        # hist_exog = windows_batch["hist_exog"]  #   [B, hist_exog_size (X), L, N]
        # futr_exog = windows_batch["futr_exog"]  #   [B, futr_exog_size (F), L + h, N]
        # stat_exog = windows_batch["stat_exog"]  #   [N, stat_exog_size (S)]
        ### ADAPTED PARSING TO MATCH PIPELINE ###
        x = windows_batch[0]['past_target'] #   [B, L, N]
        hist_exog = windows_batch[0]['past_features'] # [B, L, X]
        futr_exog = windows_batch[0]['future_features'] # [B, L+h, F]
        stat_raw = windows_batch[0]['static'] # [B, S]
        ########################################
        batch_size, input_size = x.shape[:2]

        # Apply revin to x
        if self.revin:
            x = self.norm(x, mode="norm")  #   [B, L, N] -> [B, L, N]
        
        # Apply revin to historical exogenous
        if self.revin_exo and self.hist_exog_size > 0:
            hist_exog = self.norm_hist_exog(hist_exog, mode="norm")  #   [B, L, X] -> [B, L, X]

        ### commented out
        # Add channel dimension to x
        # x = x.unsqueeze(1)  #   [B, L, N] -> [B, 1, L, N]

        # Concatenate x with historical exogenous
        if self.hist_exog_size > 0:
            ### change cat dimension
            # x = torch.cat(
            #     (x, hist_exog), dim=1
            # )  #   [B, 1, L, N] + [B, X, L, N] -> [B, 1 + X, L, N]
            x = torch.cat(
                (x, hist_exog), dim=2
            )  #   [B, L, N] + [B, L, X] -> [B, L, N + X]

        # Apply revin to future exogenous (full sequence including forecast horizon)
        if self.revin_exo and self.futr_exog_size > 0:
            futr_exog = self.norm_futr_exog(futr_exog, mode="norm")  #   [B, L+h, F] -> [B, L+h, F]
        
        # Concatenate x with future exogenous of input sequence
        if self.futr_exog_size > 0:
            ### change cat dimension
            # futr_exog_hist = futr_exog[
            #     :, :, :input_size
            # ]  #   [B, F, L + h, N] -> [B, F, L, N]
            # x = torch.cat(
            #     (x, futr_exog_hist), dim=1
            # )  #   [B, 1 + X, L, N] + [B, F, L, N] -> [B, 1 + X + F, L, N]
            futr_exog_hist = futr_exog[
                :, :input_size, :
            ]  #   [B, L + h, F] -> [B, L, F]
            x = torch.cat(
                (x, futr_exog_hist), dim=2
            )  #   [B, L, N + X] + [B, L, F] -> [B, L, N + X + F]
        
        # Temporal projection & feature mixing of x
        ### adjust permutation
        # x = x.permute(0, 1, 3, 2)  #   [B, 1 + X + F, L, N] -> [B, 1 + X + F, N, L]
        x = x.permute(0, 2, 1)  #   [B, L, N + X + F] -> [B, N + X + F, L]
        x = self.temporal_projection(
            x
        )  #   [B, 1 + X + F, N, L] -> [B, 1 + X + F, N, h]
        ### adjust permutation and comment reshape
        # x = x.permute(0, 3, 1, 2)  #   [B, 1 + X + F, N, h] -> [B, h, 1 + X + F, N]
        # x = x.reshape(
        #     batch_size, self.h, -1
        # )  #   [B, h, 1 + X + F, N] -> [B, h, (1 + X + F) * N]
        x = x.permute(0, 2, 1)  #   [B, N + X + F, h] -> [B, h,N + X + F]
        ###
        x = self.feature_mixer_hist(x)  #   [B, h, (1 + X + F) * N] -> [B, h, ff_dim]

        # Concatenate x with future exogenous of output horizon
        if self.futr_exog_size > 0:
            ### adjust future exogenous handling
            # x_futr = futr_exog[:, :, input_size:]  #   [B, F, L + h, N] -> [B, F, h, N]
            # x_futr = x_futr.permute(0, 2, 1, 3)  #   [B, F, h, N] -> [B, h, F, N]
            # x_futr = x_futr.reshape(
            #     batch_size, self.h, -1
            # )  #   [B, h, N, F] -> [B, h, N * F]
            x_futr = futr_exog[:, input_size:, :]  #   [B, L + h, F] -> [B, h, F]
            ###
            x_futr = self.feature_mixer_futr(
                x_futr
            )  #   [B, h, N * F] -> [B, h, ff_dim]
            x = torch.cat(
                (x, x_futr), dim=2
            )  #   [B, h, ff_dim] + [B, h, ff_dim] -> [B, h, 2 * ff_dim]

        # Concatenate x with static exogenous
        if self.stat_exog_size > 0:
            ### adjust static features handling
            # stat_exog = stat_exog.reshape(-1)  #   [N, S] -> [N * S]
            # stat_exog = (
            #     stat_exog.unsqueeze(0).unsqueeze(1).repeat(batch_size, self.h, 1)
            # )  #   [N * S] -> [B, h, N * S]
            stat_exog = self.static_embedding(stat_raw)  # [B, E]
            stat_exog = stat_exog.unsqueeze(1).repeat(1, self.h, 1)  # [B, S] -> [B, h, S]
            ####
            x_stat = self.feature_mixer_stat(
                stat_exog
            )  #   [B, h, N * S] -> [B, h, ff_dim]
            x = torch.cat(
                (x, x_stat), dim=2
            )  #   [B, h, 2 * ff_dim] + [B, h, ff_dim] -> [B, h, 3 * ff_dim]

        # First mixing layer
        x = self.first_mixing(x)  #   [B, h, 3 * ff_dim] -> [B, h, ff_dim]

        # N blocks of mixing layers
        if self.stat_exog_size > 0:
            x, _ = self.mixing_block(
                (x, stat_exog)
            )  #   [B, h, ff_dim], [B, h, N * S] -> [B, h, ff_dim]
        else:
            x = self.mixing_block(x)  #   [B, h, ff_dim] -> [B, h, ff_dim]

        # Fully connected output layer
        forecast = self.out(x)  #   [B, h, ff_dim] -> [B, h, N * n_outputs]

        # Reverse Instance Normalization on output
        if self.revin:
            forecast = forecast.reshape(
                batch_size, self.h * self.loss.outputsize_multiplier, -1
            )  #   [B, h, N * n_outputs] -> [B, h * n_outputs, N]
            forecast = self.norm(forecast, "denorm")
            forecast = forecast.reshape(
                batch_size, self.h, -1
            )  #   [B, h * n_outputs, N] -> [B, h, n_outputs * N]

        return forecast

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt


### Include RevINMultivariate module from common._modules used in TSMixerx ###
class RevINMultivariate(nn.Module):  
    """Reversible Instance Normalization for multivariate time series models.  

    Normalizes multivariate time series data using batch statistics computed across   
    the time dimension. The normalization can be reversed after model predictions to   
    restore the original scale. Optionally includes learnable affine parameters for   
    additional transformation flexibility.  

    Args:  
        num_features (int): The number of features or channels in the time series.  
        eps (float): A value added for numerical stability.
        affine (bool): If True, RevINMultivariate has learnable affine parameters   
            (weight and bias).
        subtract_last (bool): Not used in this implementation (kept for API compatibility).  
        non_norm (bool): Not used in this implementation (kept for API compatibility).

    Returns:
        (torch.Tensor): Normalized tensor (if mode="norm") or denormalized tensor
            (if mode="denorm") of the same shape as the input [batch, seq_len, num_features].

    Notes:
        - The forward method requires a mode parameter: "norm" for normalization or
          "denorm" for denormalization.
        - Batch statistics (mean and std) are computed across axis=1 (time dimension).
        - If affine=True, learnable parameters have shape [1, 1, num_features].
        - Used in multivariate models like TSMixer, TSMixerx, and RMoK.
    """

    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones((1, 1, self.num_features)))
        self.affine_bias = nn.Parameter(torch.zeros((1, 1, self.num_features)))

    def _normalize(self, x):
        # Batch statistics
        self.batch_mean = torch.mean(x, axis=1, keepdim=True).detach()
        self.batch_std = torch.sqrt(
            torch.var(x, axis=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

        # Instance normalization
        x = x - self.batch_mean
        x = x / self.batch_std

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        # Reverse the normalization
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight

        x = x * self.batch_std
        x = x + self.batch_mean

        return x
