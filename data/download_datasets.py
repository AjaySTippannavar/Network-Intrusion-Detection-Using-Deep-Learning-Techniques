import os, argparse, shutil, requests, zipfile, tarfile, gzip
from tqdm import tqdm

DATASETS = {
    'KDD': {
        'url': 'https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz',
        'file': 'kddcup.data_10_percent.gz',
        'note': 'KDD Cup 99 dataset (10% subset from UCI)'
    },
    'NSL': {
        'url': 'https://github.com/defcom17/NSL_KDD/archive/refs/heads/master.zip',
        'file': 'NSL-KDD-master.zip',
        'note': 'NSL-KDD GitHub mirror (works fine)'
    },
    'UNSW': {
        'url': 'https://www.dropbox.com/scl/fi/5cg9vdmrx5o4m3tz7v1ul/UNSW_NB15_training-set.csv?rlkey=jsntrj3uxym30dpnc0l02y4gy&dl=1',
        'file': 'UNSW_NB15_training-set.csv',
        'note': 'UNSW-NB15 training set (Dropbox mirror of Kaggle copy)'
    },
    'CICIDS': {
        'url': None,
        'file': None,
        'note': 'CICIDS2017 is large; please download manually from https://www.unb.ca/cic/datasets/ids-2017.html'
    }
}

def download_file(url, out_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(out_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(out_path)) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def extract_file(path, out_dir):
    if path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(out_dir)
            print(f"✅ Extracted ZIP into {out_dir}")
    elif path.endswith('.gz') and not path.endswith('.tar.gz'):
        out_file = os.path.join(out_dir, os.path.basename(path)[:-3])
        with gzip.open(path, 'rb') as f_in, open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            print(f"✅ Extracted GZ to {out_file}")
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        with tarfile.open(path, 'r:gz') as t:
            t.extractall(out_dir)
            print(f"✅ Extracted TAR.GZ into {out_dir}")
    else:
        print(f"ℹ️ Skipping extraction for plain file: {os.path.basename(path)}")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main(dataset):
    if dataset not in DATASETS:
        print(f"❌ Unknown dataset: {dataset}")
        return
    info = DATASETS[dataset]
    if not info.get('url'):
        print(f"⚠️ Manual download required: {info.get('note')}")
        return
    out_dir = os.path.join('data','datasets',dataset)
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, info['file'])
    print(f"📦 Dataset: {dataset}\n🌐 Source: {info['url']}\n💾 Destination: {out_file}")
    if os.path.exists(out_file):
        print(f"✅ Already downloaded: {out_file}")
    else:
        try:
            download_file(info['url'], out_file)
            print(f"✅ Download complete: {out_file}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return
    try:
        extract_file(out_file, out_dir)
    except Exception as e:
        print(f"⚠️ Extraction skipped or failed: {e}")
    print(f"🎉 Done for {dataset}\n{'-'*60}")

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(DATASETS.keys()))
    p.add_argument('--all', action='store_true')
    a=p.parse_args()
    if a.all:
        for k in DATASETS: main(k)
    else:
        main(a.dataset)
