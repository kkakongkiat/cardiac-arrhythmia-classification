"""Four-class notebook -> export -> app checks using synthetic local WFDB data."""
import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import wfdb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from program import inference as app

ROOT = Path(__file__).resolve().parents[1]


def definitions(filename, names, namespace):
    nodes = []
    for cell in json.loads((ROOT / filename).read_text(encoding='utf-8'))['cells']:
        if cell['cell_type'] == 'code':
            nodes.extend(n for n in ast.parse(''.join(cell['source'])).body
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), filename, 'exec'), namespace)
    return namespace


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def export_function(name='export_bundle'):
    import csv
    import hashlib
    import shutil
    import tempfile
    import math
    namespace = dict(Path=Path, csv=csv, hashlib=hashlib, json=json,
                     shutil=shutil, tempfile=tempfile, math=math, ROOT=ROOT)
    return definitions('model/10_Export_model.ipynb', {'read_json', 'sha256', 'export_bundle', 'export_all_models'}, namespace)[name]


@pytest.fixture
def record(tmp_path):
    signal = np.sin(np.arange(1200) / 20.) + .1 * np.cos(np.arange(1200) / 7.)
    wfdb.wrsamp('100', fs=360, units=['mV'], sig_name=['MLII'],
                p_signal=signal[:, None], fmt=['16'], write_dir=str(tmp_path))
    # Includes an out-of-scope Q and two boundary beats. All contribute to RR.
    wfdb.wrann('100', 'atr', sample=np.array([20, 150, 400, 650, 900, 1150]),
               symbol=['N', 'N', 'Q', 'V', 'F', 'S'], write_dir=str(tmp_path))
    return tmp_path / '100'


def make_bundle(root, kind, record):
    signal, _ = app.read_record(record)
    samples, _ = app.read_beats(record)
    waves, valid = app.segment(signal, samples)
    rr = app.rr_features(samples)[valid]
    values, names = app.extract_features(waves, rr)
    ns = definitions('dataset/4_Feature_engineering.ipynb', {'fit_transforms', 'transform_partition'},
                     {**vars(app), 'SimpleImputer': SimpleImputer, 'RobustScaler': RobustScaler})
    raw = {'wave': waves, 'rr': rr, 'x': values}
    transforms = ns['fit_transforms'](raw)
    part = ns['transform_partition'](raw, transforms)
    parameters = {'base_channels': 4, 'stages': 1, 'blocks_per_stage': 1, 'kernel': 5,
                  'lstm_hidden': 4, 'lstm_layers': 1, 'dropout': .1,
                  'rr_hidden': 4, 'head_hidden': 8}
    folder = root / 'model/artifacts' / kind / 'example/final'
    folder.mkdir(parents=True)
    y = np.arange(4)
    if kind == 'rf':
        model = RandomForestClassifier(n_estimators=3, random_state=42).fit(part['x'], y)
        filename = 'model.joblib'
        joblib.dump(model, folder / filename)
    elif kind == 'xgb':
        model = xgb.train({'objective': 'multi:softprob', 'num_class': 4, 'max_depth': 2, 'nthread': 1},
                          xgb.DMatrix(part['x'], label=y), num_boost_round=2)
        filename = 'model.ubj'
        model.save_model(folder / filename)
    else:
        notebook = '7_CNN' if kind == 'cnn' else '8_CNN_BiLSTM'
        ns = definitions(f'model/{notebook}.ipynb', {'ResidualBlock', 'ECGNetwork'}, dict(vars(app)))
        torch.manual_seed(42)
        model = ns['ECGNetwork'](kind, parameters, 24).eval()
        filename = 'model.pt'
        torch.save(model.state_dict(), folder / filename)
    # Compare notebook/app serialization on the same device, avoiding CPU/CUDA
    # kernel precision differences on the user's GPU-enabled environment.
    device = 'cuda' if kind in ['cnn', 'cnn_bilstm'] and torch.cuda.is_available() else 'cpu'
    if kind in ['cnn', 'cnn_bilstm']:
        model = model.to(device)
    expected = app.probabilities(model, kind, part, device)
    config = json.loads((ROOT / 'benchmark_config.json').read_text(encoding='utf-8'))
    write_json(root / 'benchmark_config.json', config)
    data = root / 'dataset/processed'
    state = {'config_hash': app.sha256(root / 'benchmark_config.json'), 'classes': app.CLASSES,
             'shape': [2, 271], 'offline': True, 'rr_names': app.RR_NAMES}
    schema = {'names': names, 'rr_names': app.RR_NAMES, 'wavelet': 'db4', 'wavelet_packet_level': 4}
    write_json(data / 'preprocessing.json', state)
    write_json(data / 'feature_schema.json', schema)
    write_json(data / 'prepared.json', {'config_hash': state['config_hash'], 'files': {
        f: app.sha256(data / f) for f in ['preprocessing.json', 'feature_schema.json']}})
    info = {'model': kind, 'classes': app.CLASSES, 'parameters': parameters,
            'context_dim': 24, 'prepared_hash': app.sha256(data / 'prepared.json')}
    write_json(folder / 'info.json', info)
    joblib.dump(transforms, folder / 'transforms.joblib')
    report = root / 'model/reports/final'
    report.mkdir(parents=True)
    kinds = [kind] + [k for k in ['rf', 'xgb', 'cnn', 'cnn_bilstm'] if k != kind]
    pd.DataFrame({'model': kinds, 'macro_f1': [.8, .7, .6, .5]}).to_csv(report / 'comparison.csv', index=False)
    snapshot = {'path': f'{kind}/example/final', 'files': {
        f: app.sha256(folder / f) for f in [filename, 'info.json', 'transforms.joblib']}}
    write_json(report / 'evaluation.json', {'classes': app.CLASSES, 'protocol': config['protocol'],
        'ranking_metric': 'test_macro_f1', 'hyperparameters_selected_on': 'validation',
        'prepared_hash': info['prepared_hash'], 'config_hash': state['config_hash'],
        'comparison_hash': app.sha256(report / 'comparison.csv'), 'benchmark_top_model': kind,
        'artifact_snapshots': {k: snapshot for k in kinds}})
    bundle, _ = export_function()(root)
    return bundle, expected, folder


@pytest.mark.parametrize('kind', ['rf', 'xgb', 'cnn', 'cnn_bilstm'])
def test_notebook_model_export_and_prediction(tmp_path, record, kind):
    bundle, expected, _ = make_bundle(tmp_path / 'project', kind, record)
    progress = []
    actual = app.predict_record(bundle, record, progress=lambda *event: progress.append(event))
    scores = actual['predictions'][[f'score_{c}' for c in app.CLASSES]].to_numpy()
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)
    assert actual['predictions'].r_sample.tolist() == [150, 400, 650, 900]
    assert scores.shape == (4, 4)
    assert len(progress) == 12


def test_preprocessing_matches_notebook_including_q_context(record):
    ns = definitions('dataset/3_Preprocessing.ipynb', {'filter_ecg', 'rr_features', 'segment_record'},
                     {**vars(app), 'RAW': record.parent})
    samples, symbols = app.read_beats(record)
    target = np.array([s in app.LABEL_MAP for s in symbols])
    meta = pd.DataFrame({'r_sample': samples[target], 'beat_id': [str(s) for s in samples[target]], 'record_id': '100'})
    selected, expected_waves, expected_rr, _ = ns['segment_record']('100', meta)
    signal, _ = app.read_record(record)
    waves, valid = app.segment(signal, samples)
    target_valid = np.isin(samples[valid], selected.r_sample)
    np.testing.assert_array_equal(waves[target_valid], expected_waves)
    np.testing.assert_allclose(app.rr_features(samples)[valid][target_valid], expected_rr, equal_nan=True)
    features_ns = definitions('dataset/4_Feature_engineering.ipynb', {'extract_features'}, dict(vars(app)))
    expected, names = features_ns['extract_features'](expected_waves, expected_rr)
    actual, actual_names = app.extract_features(waves[target_valid], expected_rr)
    assert len(names) == 169 and actual_names == names
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_export_rejects_changed_checkpoint_and_preserves_bundle(tmp_path, record):
    root = tmp_path / 'project'
    bundle, _, folder = make_bundle(root, 'rf', record)
    original = app.sha256(bundle / 'bundle.json')
    (folder / 'model.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Evaluated artifact changed'):
        export_function()(root)
    assert app.sha256(bundle / 'bundle.json') == original


def test_bundle_rejects_changed_or_legacy_artifacts(tmp_path, record):
    bundle, _, _ = make_bundle(tmp_path / 'project', 'rf', record)
    (bundle / 'transforms.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Bundle file changed'):
        app.load_bundle(bundle)
    write_json(bundle / 'bundle.json', {'version': 1})
    with pytest.raises(ValueError, match='version 2'):
        app.load_bundle(bundle)


def test_contract_and_input_rejections():
    with pytest.raises(ValueError, match='MLII'):
        app.choose_lead(['V5'])
    for samples in [[150, 150], [400, 150], [100.5, 400.5], [-1, 500]]:
        with pytest.raises(ValueError):
            app.segment(np.ones(1000), samples)


def test_desktop_service_analysis_and_out_of_scope_truth(tmp_path, record):
    import threading
    from program.service import analyze, export_predictions
    bundle, _, _ = make_bundle(tmp_path / 'project', 'rf', record)
    result = analyze(record, bundle, lambda *event: None, threading.Event())
    frame = result['predictions']
    assert frame.loc[frame.r_sample.eq(400), 'true_class'].iloc[0] == 'Q (outside task)'
    assert not frame.loc[frame.r_sample.eq(400), 'in_benchmark_scope'].iloc[0]
    for include_truth in [False, True]:
        output = tmp_path / f'predictions_{include_truth}.csv'
        export_predictions(result, output, include_truth)
        restored = pd.read_csv(output)
        assert ('true_class' in restored) == include_truth
        assert len(restored) == result['beats']
        assert restored.model.eq('rf').all()
