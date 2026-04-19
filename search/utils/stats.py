import numpy as np


def remove_outliers(data: list[float], iqr_k: float = 1.5) -> list[float]:
    """Remove outliers using Tukey's fences (IQR method)."""
    if not data:
        return []
    arr = np.array(data)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    return list(arr[(arr >= low) & (arr <= high)])


def winrate(scores: list[float]) -> float | None:
    """Mean of scores, with IQR outlier removal for continuous values."""
    if not scores:
        return None
    return float(np.mean(scores))
