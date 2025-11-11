import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder
import tensorflow as tf
from models.attention import SimpleAttention


def safe_json(o):
    """Make numpy types JSON serializable."""
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, choices=['KDD', 'NSL', 'UNSW', 'CICIDS'])
p.add_argument('--model', required=True, choices=['snn', 'dnn', 'cnn', 'attention'])
a = p.parse_args()

proc = f'data/processed/{a.dataset}'
if not os.path.exists(proc):
    raise FileNotFoundError('Processed data not found. Run preprocessing first.')

# Load feature & labels
X_path = os.path.join(proc, 'X.csv')
y_path = os.path.join(proc, 'y.csv')
if not os.path.exists(X_path) or not os.path.exists(y_path):
    raise FileNotFoundError('X.csv or y.csv missing in processed data dir: ' + proc)

X = pd.read_csv(X_path).values
y = pd.read_csv(y_path, header=None).iloc[:, 0].astype(str).values

# Load label encoder if present, otherwise fit new one (consistent ordering)
le_path = os.path.join(proc, 'label_encoder.joblib')
if os.path.exists(le_path):
    le = joblib.load(le_path)
else:
    le = LabelEncoder().fit(y)
    joblib.dump(le, le_path)

# load model
model_path = f'saved_models/{a.dataset}_{a.model}.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError('Model not found. Train the model first: ' + model_path)

print('Loading model:', model_path)
# load model while telling Keras about the custom layer
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'SimpleAttention': SimpleAttention},
                                   compile=False)


# Predict probabilities (soft outputs) and labels
print('Running predictions on X shape:', X.shape)
probs = model.predict(X, batch_size=256)
# if model outputs a single value per sample (binary single-dim), handle accordingly
if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
    # binary decision: convert to 2-column probabilities [1-p, p]
    probs = np.stack([1 - probs.ravel(), probs.ravel()], axis=1)

preds_idx = np.argmax(probs, axis=1)
try:
    preds_str = le.inverse_transform(preds_idx)
except Exception:
    # fallback: try to build label encoder from y and map ints
    le2 = LabelEncoder().fit(y)
    preds_str = le2.inverse_transform(preds_idx)

# Classification report and confusion matrix
print('Computing classification report and confusion matrix...')
report = classification_report(y, preds_str, output_dict=True, zero_division=0)
labels = list(le.classes_)
cm = confusion_matrix(y, preds_str, labels=labels)

# Attempt ROC computation
roc_obj = None
try:
    # Binarize true labels using same classes ordering
    y_bin = label_binarize(y, classes=labels)  # shape (n_samples, n_classes)
    # Ensure probs shape aligns with classes
    if probs.shape[1] == y_bin.shape[1]:
        # micro-average by flattening (works as a quick overall ROC)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)
        roc_obj = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
    else:
        # fallback: try a per-class ROC for the first class (if available)
        if y_bin.shape[1] > 0 and probs.shape[1] > 0:
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, 0], probs[:, 0])
                roc_auc = auc(fpr, tpr)
                roc_obj = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
            except Exception:
                roc_obj = None
        else:
            roc_obj = None
except Exception as e:
    print('ROC computation failed:', str(e))
    roc_obj = None

# Load training history (if saved by train.py)
hist_file = os.path.join("saved_models", f"{a.dataset}_{a.model}_history.json")
loss_obj = None
if os.path.exists(hist_file):
    try:
        with open(hist_file, 'r') as fh:
            hist = json.load(fh)
        train_loss = hist.get("loss") or hist.get("train") or hist.get("train_loss")
        val_loss = hist.get("val_loss") or hist.get("val") or hist.get("validation_loss")
        epochs = len(train_loss) if train_loss else (hist.get("epochs") or None)
        loss_obj = {"train": train_loss, "val": val_loss, "epochs": epochs}
    except Exception as e:
        print('Failed to load history file:', e)
        loss_obj = None
else:
    # No history file found -> loss_obj stays None
    pass

# Prepare output JSON
out = {
    "report": report,
    "confusion_matrix": cm.tolist(),
    "labels": labels,
    "accuracy": report.get("accuracy", None),
    "roc": roc_obj,
    "loss": loss_obj
}

# Write to disk (same naming pattern you used before)
out_path = model_path + '_report.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print('Saved report to', out_path)
# print a short summary for quick debugging
print('Available keys in output:', list(k for k in out.keys() if out[k] is not None))
