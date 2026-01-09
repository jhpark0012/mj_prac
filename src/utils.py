from sklearn.metrics import f1_score


def compute_f1(y_true, y_pred, average="macro"):

    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu().numpy()

    return f1_score(y_true, y_pred, average=average)
