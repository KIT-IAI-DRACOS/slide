import os
import pickle
from collections.abc import Sequence

import torch

from models.basemodel import PredictionOutput


def extract_predictions(
    prediction: Sequence[PredictionOutput],
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[str]] | None,
]:
    """
    Extract the predictions (y_hat, y) from the prediction list.

    Arguments:
        prediction: List of tuples containing predictions.

    Returns:
        y_hat, y, ids: Extracted predictions, true values, and optional ids per sample.
    """
    y_hat = [pred[0] for pred in prediction]
    y = [pred[1] for pred in prediction]

    # Collect ids if present (e.g., returned by predict_step as part of x dict)
    ids_batches = []
    for pred in prediction:
        if len(pred) >= 3 and isinstance(pred[2], dict) and "id" in pred[2]:
            ids_batches.append(pred[2]["id"])

    ids = ids_batches if len(ids_batches) == len(prediction) else None
    return y_hat, y, ids


def save_predictions(
    output_dir: str | os.PathLike[str],
    y_hat: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """
    Save y_hat and y predictions to files.

    Arguments:
        output_dir: Destination directory where predictions should be stored.
        y_hat, y: Predictions and true values to save.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "y_hat.pkl"), "wb") as f:
        pickle.dump(y_hat, f)
    with open(os.path.join(output_dir, "y.pkl"), "wb") as f:
        pickle.dump(y, f)


def flatten_ids(ids_batches: Sequence[Sequence] | None) -> list[str] | None:
    """
    Flatten a list of id batches (one batch per prediction output) into a single
    list aligned with concatenated predictions.
    """
    if ids_batches is None:
        return None

    flat: list[str] = []
    for batch in ids_batches:
        if batch is None:
            return None
        flat.extend([str(i) for i in batch])
    return flat
