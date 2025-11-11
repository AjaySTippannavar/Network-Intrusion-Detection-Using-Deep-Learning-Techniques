from flask import Flask, send_file, jsonify, request
import subprocess, os, json

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    # Directly send HTML to avoid Jinja parsing errors
    return send_file('templates/index.html')

@app.route('/api/datasets')
def datasets():
    return jsonify(['KDD','NSL','UNSW','CICIDS'])

@app.route('/api/download/<dataset>', methods=['POST'])
def download(dataset):
    if dataset not in ['KDD','NSL','UNSW','CICIDS']:
        return jsonify({'error':'unknown dataset'}),400
    try:
        subprocess.run(['python','data/download_datasets.py','--dataset',dataset], check=True)
        return jsonify({'status':'downloaded'})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/train/<dataset>/<model>', methods=['POST'])
def train(dataset, model):
    try:
        subprocess.run(['python','train.py','--dataset',dataset,'--model',model,'--epochs','5'], check=True)
        subprocess.run(['python','evaluate.py','--dataset',dataset,'--model',model], check=True)
        return jsonify({'status':'trained_and_evaluated'})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/report/<dataset>/<model>')
def report(dataset, model):
    path = os.path.join('saved_models', f'{dataset}_{model}.h5_report.json')
    if os.path.exists(path):
        with open(path) as f: return jsonify(json.load(f))
    return jsonify({'error':'report not found'}),404

if __name__=='__main__':
    print('Starting dashboard at http://127.0.0.1:8050')
    app.run(host='0.0.0.0', port=8050, debug=True)
