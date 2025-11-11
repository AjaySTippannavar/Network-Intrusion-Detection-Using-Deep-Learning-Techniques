import os, argparse, joblib, subprocess
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# === Import models ===
from models.shallow import build_shallow
from models.deep import build_deep
from models.cnn import build_cnn
from models.attention import build_attention

# === Parse CLI arguments ===
p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, choices=['KDD', 'NSL', 'UNSW', 'CICIDS'])
p.add_argument('--model', required=True, choices=['snn', 'dnn', 'cnn', 'attention'])
p.add_argument('--epochs', type=int, default=5)
a = p.parse_args()

# === Ensure processed dataset ===
def ensure_processed(ds):
    """Ensure processed dataset exists; if not, auto-preprocess."""
    proc_dir = os.path.join('data', 'processed', ds)
    os.makedirs(proc_dir, exist_ok=True)

    X_path = os.path.join(proc_dir, 'X.csv')
    y_path = os.path.join(proc_dir, 'y.csv')
    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"✅ Processed data found for {ds}")
        return

    print(f"Processed data not found for {ds}")
    print(f"Attempting to locate raw dataset and preprocess...")

    # Locate downloaded dataset files
    raw_dir = os.path.join('data', 'datasets', ds)
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"No dataset directory found for {ds}")

    # Find any suitable file (gz, csv, txt)
    candidates = []
    for ext in ('.csv', '.txt', '.gz', ''):
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if ext in f:
                    candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError('No raw dataset file found for ' + ds)

    first = candidates[0]
    print(f"Found raw dataset: {first}")

    # Run preprocessing script
    cmd = ['python', 'data/preprocess_all.py', '--input', first, '--out', proc_dir]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# === Load processed data ===
ensure_processed(a.dataset)
out = f'data/processed/{a.dataset}'

X = pd.read_csv(os.path.join(out, 'X.csv')).values
y = pd.read_csv(os.path.join(out, 'y.csv'), header=None).iloc[:, 0].astype(str).values

# Label encoding
le_path = os.path.join(out, 'label_encoder.joblib')
if os.path.exists(le_path):
    le = joblib.load(le_path)
else:
    le = LabelEncoder().fit(y)
    joblib.dump(le, le_path)

le2 = LabelEncoder()
y_enc = le2.fit_transform(y)
joblib.dump(le2, le_path)

# === Build model ===
input_dim = X.shape[1]
num_classes = len(set(y_enc))

if a.model == 'snn':
    model = build_shallow(input_dim, num_classes)
elif a.model == 'dnn':
    model = build_deep(input_dim, num_classes)
elif a.model == 'cnn':
    model = build_cnn(input_dim, num_classes)
elif a.model == 'attention':
    model = build_attention(input_dim, num_classes)

# === Split data ===
Xtr, Xva, ytr, yva = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

# === Train ===
os.makedirs("saved_models", exist_ok=True)
ckpt = f'saved_models/{a.dataset}_{a.model}.h5'
cb = [tf.keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True, monitor='val_accuracy', verbose=1)]

print(f"🚀 Starting training: {a.dataset} — {a.model} for {a.epochs} epochs")

history = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=a.epochs,
    batch_size=32,
    callbacks=cb,
    verbose=2
)

print(f"✅ Training completed. Best model saved at: {ckpt}")

# === Save training history ===
hist_obj = {
    "loss": history.history.get("loss", []),
    "val_loss": history.history.get("val_loss", [])
}

hist_file = os.path.join("saved_models", f"{a.dataset}_{a.model}_history.json")
with open(hist_file, "w") as fh:
    import json
    json.dump(hist_obj, fh, indent=2)

print(f" Saved training history to: {hist_file}")
print("🎉 Done.")
