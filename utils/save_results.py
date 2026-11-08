from collections.abc import Sequence

import torch
from omegaconf import DictConfig

from models.basemodel import PredictionOutput
from utils.logging_utils.metrics import log_metrics
from utils.main_utils.run_context import RunContext
from utils.saving_utils.data_handling import extract_predictions, flatten_ids, save_predictions


def save_results(
    config: DictConfig,
    prediction: Sequence[PredictionOutput],
    run_context: RunContext,
    name: str = "",
) -> None:
    """
    Save model predictions and metrics.

    Arguments:
        config: Configuration object containing model, sample, and output settings.
        prediction: List of tuples where each tuple contains (y_hat, y) and possibly more.
        run_context: Shared run context containing the resolved output directory.
        name: used as a suffix in logging to wandb (e.g. "-end" if training end is logged in addition to best model at checkpoint)
    """
    y_hat, y, ids = extract_predictions(prediction)
    ids_flat = flatten_ids(ids)

    # reshape tensors
    y_hat = torch.cat(y_hat)
    y = torch.cat(y)
    
    # save predictions
    if config.save_predictions:
        save_predictions(run_context.run_dir, y_hat, y)
    
    # calculate and log metrics
    log_metrics(config, y, y_hat, name, ids_flat)
