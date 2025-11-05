import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_metrics(y_true: np.ndarray | list, y_pred: np.ndarray | list, threshold: float = 0.5) -> dict[str, float]:
    """
    여러 이진 분류 지표 반환

    Args:
        y_true (np.ndarray or list): 정답 레이블 (0/1 또는 True/False)
        y_pred (np.ndarray or list): 예측값 (확률 or 0/1)
        threshold (float): 이진 분류일 경우 확률 예측 시 임계값

    Returns:
        dict: 다양한 평가 지표
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If prediction is float, apply threshold to get label
    if y_pred.dtype.kind in "fc":
        y_pred_label = (y_pred >= threshold).astype(int)
    else:
        y_pred_label = y_pred.astype(int)

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred_label)
    metrics["f1"] = f1_score(y_true, y_pred_label)
    metrics["roc_auc"] = roc_auc_score(y_true, y_pred)

    return metrics
