import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.exceptions import DataValidationError


def regression_metrics(y_true, y_pred) -> dict:
    """Evaluate one-dimensional price predictions in original units."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    # Accept either (samples,) or (samples, 1).
    if actual.ndim == 2 and actual.shape[1] == 1:
        actual = actual[:, 0]

    if predicted.ndim == 2 and predicted.shape[1] == 1:
        predicted = predicted[:, 0]

    if actual.ndim != 1 or predicted.ndim != 1:
        raise DataValidationError("Expected one target per sample.")

    if actual.shape != predicted.shape or actual.size < 2:
        raise DataValidationError(
            "Actual and predicted arrays must match and contain "
            "at least two observations."
        )

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise DataValidationError("Metrics require finite values.")

    if (actual <= 0).any():
        raise DataValidationError(
            "This price evaluation expects positive actual prices."
        )

    if np.all(actual == actual[0]):
        raise DataValidationError(
            "R² is undefined for constant actual targets."
        )

    mse = float(mean_squared_error(actual, predicted))

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
        "MAPE_percent": float(
            100 * mean_absolute_percentage_error(actual, predicted)
        ),
        "R2": float(r2_score(actual, predicted)),
    }