import numpy as np
import pandas as pd
import torch
import wandb


def log_metrics(config, y, y_hat, name, sample_ids=None):
    """
    Calculate and log metrics.

    Arguments:
        config: Configuration object containing model information.
        y, y_hat: True values and predictions.
        name: Name of the model or experiment for logging.
        sample_ids: Optional list of ids aligned with the first dimension of y/y_hat.
    """
    device = "cpu"
    
    y = y.to(device).numpy() if torch.is_tensor(y) else np.asarray(y)
    y_hat = y_hat.to(device).numpy() if torch.is_tensor(y_hat) else np.asarray(y_hat)
    if y.ndim == 2:
        y = y[:, :, None]
    if y_hat.ndim == 2:
        y_hat = y_hat[:, :, None]
    ids = np.asarray(sample_ids) if sample_ids is not None else None
    if ids is not None and len(ids) != y.shape[0]:
        ids = None
    if ids is not None:
        ids = ids.astype(str)

    if config.model.loss_function == "logmse":
        rmsle = np.sqrt(np.mean((y_hat - y) ** 2))
        wandb.log({f"RMSLE{name}": rmsle})
        y_hat = np.expm1(y_hat)  # invert log1p
        y = np.expm1(y)


    metrics = {
        "MSE": lambda y, y_hat, axis: np.mean((y_hat - y) ** 2, axis=axis),
        "MAE": lambda y, y_hat, axis: np.mean(np.abs(y_hat - y), axis=axis),
        "SMAPE": lambda y, y_hat, axis: np.mean(
            2 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)), axis=axis
        ),
        "p90": lambda y, y_hat, axis: np.mean(
            np.maximum(0.9 * (y - y_hat), 0.1 * (y_hat - y)), axis=axis
        ),
        "p10": lambda y, y_hat, axis: np.mean(
            np.maximum(0.1 * (y - y_hat), 0.9 * (y_hat - y)), axis=axis
        ),
    }

    for metric_name, metric_fn in metrics.items():
        log_metric_over_time(metric_name, metric_fn(y, y_hat, (0, 2)), name)

    if ids is not None:
        # Compute per-id summary for core metrics
        sample_mse = np.mean((y_hat - y) ** 2, axis=(1, 2))
        sample_mae = np.mean(np.abs(y_hat - y), axis=(1, 2))
        df = pd.DataFrame({"timeseries_id": ids, "mse": sample_mse, "mae": sample_mae})
        per_id = df.groupby("timeseries_id").mean()
        for metric_name in ["mse", "mae"]:
            table = wandb.Table(
                data=[[idx, val] for idx, val in per_id[metric_name].items()],
                columns=["timeseries_id", f"{metric_name}_by_id{name}"],
            )
            wandb.log(
                {
                    f"{metric_name}_by_id_plot{name}": wandb.plot.bar(
                        table, "timeseries_id", f"{metric_name}_by_id{name}", title=f"{metric_name.upper()} by id{name}"
                    )
                }
            )


    if y.shape[2] > 1:
        mse_by_series = np.mean((y_hat - y) ** 2, axis=(0, 1))
        table = wandb.Table(
            data=[[i, r] for i, r in enumerate(mse_by_series)],
            columns=["series", f"mse_by_series{name}"],
        )
        wandb.log(
            {
                "mse_by_series_plot": wandb.plot.bar(
                    table, "series", f"mse_by_series{name}", title=f"MSE by Series{name}"
                )
            }
        )

    log_sample_mse(y, y_hat, name)


def log_metric_over_time(metric_name, metric_values, name):
    """
    Log a metric over time with WandB.

    Arguments:
        metric_name: Name of the metric (RMSE, MAE, etc.).
        metric_values: Calculated metric values.
        name: Name of the model or experiment for logging.
    """
    table = wandb.Table(
        data=[[i, r] for i, r in enumerate(metric_values)],
        columns=["time", f"{metric_name}{name}"],
    )
    wandb.log(
        {
            f"{metric_name}_plot": wandb.plot.line(
                table, "time", f"{metric_name}{name}", title=f"{metric_name} {name}"
            )
        }
    )
    wandb.log({f"{metric_name}{name}": np.mean(metric_values)})


def log_sample_mse(y, y_hat, name):
    """
    Log sample-wise MSE.

    Arguments:
        y, y_hat: True values and predictions.
        name: Name of the model or experiment for logging.
    """
    sample_mse = np.mean((y_hat - y) ** 2, axis=1)

    table = wandb.Table(
        data=[[i, r] for i, r in enumerate(sample_mse.ravel())],
        columns=["sample", f"sample_mse{name}"],
    )
    wandb.log(
        {
            "sample_mse_plot": wandb.plot.line(
                table, "sample", f"sample_mse{name}", title=f"Sample MSE {name}"
            )
        }
    )
