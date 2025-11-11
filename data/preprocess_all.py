import os, argparse, pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess(input_csv, out_dir):
    df = pd.read_csv(input_csv, header=None)
    df = df.drop_duplicates()
    df.columns = list(range(df.shape[1]))
    y = df.iloc[:, -1].astype(str)
    X = df.iloc[:, :-1].copy()
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder(); X[col] = le.fit_transform(X[col].astype(str))
    sc = MinMaxScaler(); Xs = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    os.makedirs(out_dir, exist_ok=True)
    Xs.to_csv(os.path.join(out_dir,'X.csv'), index=False)
    y.to_csv(os.path.join(out_dir,'y.csv'), index=False, header=False)
    joblib.dump(sc, os.path.join(out_dir,'scaler.joblib'))
    joblib.dump(LabelEncoder().fit(y), os.path.join(out_dir,'label_encoder.joblib'))
    print('Saved processed to', out_dir)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    a=p.parse_args(); preprocess(a.input, a.out)
