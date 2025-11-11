
from sklearn.metrics import roc_curve, auc
import numpy as np

def generate_extra_charts(model, X_test, y_test, history=None):
    results = {}

    # --- ROC curve ---
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] > 1:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                results["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
        elif hasattr(model, "predict"):
            y_prob = model.predict(X_test)
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                results["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
    except Exception as e:
        print("ROC generation failed:", e)

    # --- Training/Validation loss ---
    if history is not None and hasattr(history, "history"):
        results["loss_curve"] = {
            "epochs": list(range(1, len(history.history["loss"]) + 1)),
            "train_loss": history.history["loss"],
            "val_loss": history.history.get("val_loss", [])
        }

    return results
