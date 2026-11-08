from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

import lightning.pytorch as pl
import torch
import torch.nn as nn

LossName: TypeAlias = Literal["mse", "mae", "logmse"]
class ModelInputs(TypedDict):
      past_target: torch.Tensor
      past_features: NotRequired[torch.Tensor]
      future_features: NotRequired[torch.Tensor]
      static: NotRequired[torch.Tensor]
      id: NotRequired[list[str]]

ModelBatch: TypeAlias = tuple[ModelInputs, torch.Tensor]
PredictionOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor, ModelInputs]

class BaseModel(pl.LightningModule):
    """
    The base model class for all models.
    """

    def __init__(self, loss_function: LossName, **kwargs: Any) -> None:
        """Initialize the model with input dimensions and loss function."""
        super().__init__()
        self.loss_function = self._get_loss_function(loss_function, **kwargs)
        self.inverse_transform = kwargs.get("inverse_transform", False)

    def training_step(self, batch: ModelBatch, batch_idx: int) -> torch.Tensor:
        """Perform one training step."""
        _x, y, *_ = batch
        y_hat = self(batch)
        loss = self.loss_function(torch.squeeze(y), torch.squeeze(y_hat))

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

    def validation_step(self, batch: ModelBatch, batch_idx: int) -> torch.Tensor:
        """Perform one validation step."""
        _x, y, *_ = batch

        y_hat = self(batch)

        loss = self.loss_function(torch.squeeze(y), torch.squeeze(y_hat))

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=y.size(0),
        )
        return loss

    def test_step(self, batch: ModelBatch, batch_idx: int) -> None:
        """Perform one test step."""
        _x, y, *_ = batch

        y_hat = self(batch)    
        
        loss = self.loss_function(torch.squeeze(y), torch.squeeze(y_hat))

        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=y.size(0),
        )

    def predict_step(self, batch: ModelBatch, batch_idx: int) -> PredictionOutput:
        x, y, *_ = batch  # x is dict, y is [B, y_size, 1]
        y_hat = self(batch)  # [B, y_size, 1]

        if not self.inverse_transform:
            return y_hat, y, x

        scaler_cfg = self.trainer.datamodule.target_scaler

        # per-id scalers
        if isinstance(scaler_cfg, dict):
            ids = x["id"]  # list of length B, e.g. ["id_1", "id_2", ...]

            B = y.shape[0]
            y_list = []
            y_hat_list = []
            past_list = []

            for i in range(B):
                scaler = scaler_cfg[ids[i]]  # StandardScaler for this id

                y_list.append(scaler.inverse_transform(y[i]))  # [y_size, 1]
                y_hat_list.append(scaler.inverse_transform(y_hat[i]))  # [y_size, 1]
                past_list.append(scaler.inverse_transform(x["past_target"][i]))  # [x_size, 1]

            y = torch.stack(y_list, dim=0)  # [B, y_size, 1]
            y_hat = torch.stack(y_hat_list, dim=0)  # [B, y_size, 1]
            x["past_target"] = torch.stack(past_list, 0)  # [B, x_size, 1]

        # global scaler
        else:
            scaler = scaler_cfg

            y = scaler.inverse_transform(y)  # [B, y_size, 1]
            y_hat = scaler.inverse_transform(y_hat)  # [B, y_size, 1]
            x["past_target"] = scaler.inverse_transform(x["past_target"])  # [B, x_size, 1]

        return y_hat, y, x

    def _get_loss_function(self, loss_function: LossName, **kwargs: Any) -> nn.Module:
        """
        Retrieve the appropriate loss function based on the provided loss name.
        """
        loss_functions = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "logmse": nn.MSELoss(),
        }

        if loss_function not in loss_functions:
            raise NotImplementedError(
                f"Loss function '{loss_function}' not implemented. Choose from {list(loss_functions.keys())}"
            )

        chosen_loss_function = loss_functions[loss_function]

        if not hasattr(chosen_loss_function, "outputsize"):
            chosen_loss_function.outputsize = 1

        return chosen_loss_function
