"""Offline four-class inference; numerical definitions mirror notebooks 1?9.

Bundles contain the final model's fitted transforms and protocol snapshot.
Training remains in notebooks; importing this module never executes them.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
import pandas as pd
import joblib
import wfdb
import pywt
import xgboost as xgb
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset/raw/mit-bih-arrhythmia-database-1.0.0'
ARTIFACTS = ROOT / 'model/artifacts'
CLASSES = ['N', 'S', 'V', 'F']
N_CLASSES = len(CLASSES)
FS, LEFT, RIGHT = 360, 90, 180
WAVE_LENGTH = LEFT + RIGHT + 1
ALL_SYMBOLS = ['NLRej', 'AaJS', 'VE', 'F', '/fQ']
ALL_LABEL_MAP = {s: i for i, symbols in enumerate(ALL_SYMBOLS) for s in symbols}
LABEL_MAP = {s: i for s, i in ALL_LABEL_MAP.items() if i < N_CLASSES}
SUPPORTED_PROTOCOL = 'de_chazal_2004_subject_corrected_four_class_v2'


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'ไม่พบ {path}; รัน notebook ขั้นก่อนหน้าก่อน')
    return json.loads(path.read_text(encoding='utf-8'))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def transform_partition(part, transforms):
    result = dict(part)
    result['x'] = transforms['feature_imputer'].transform(part['x']).astype(np.float32)
    filled = transforms['rr_imputer'].transform(part['rr'])
    scaled = transforms['rr_scaler'].transform(filled)
    # Clipping bounds are fixed protocol constants; missing flags keep edge information.
    result['context'] = np.column_stack([np.clip(scaled, -10., 10.),
                                        ~np.isfinite(part['rr'])]).astype(np.float32)
    if not np.isfinite(result['x']).all() or not np.isfinite(result['context']).all():
        raise ValueError('Imputation/scaling produced nonfinite input')
    return result

def filter_ecg(signal):
    signal = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError('ECG contains NaN/Inf')
    # de Chazal: consecutive 200 ms / 600 ms median filters for baseline removal.
    # Adaptation: zero-phase Butterworth low-pass 35 Hz replaces the paper's 12-tap FIR.
    baseline = median_filter(median_filter(signal, size=73, mode='nearest'), size=217, mode='nearest')
    sos = butter(4, 35., btype='lowpass', fs=FS, output='sos')
    return sosfiltfilt(sos, signal - baseline)

RR_NAMES = ['rr_pre', 'rr_post', 'rr_global_mean', 'rr_local_mean', 'rr_pre_local',
            'rr_post_local', 'rr_pre_global', 'rr_post_global', 'rr_pre_post_ratio',
            'rr_delta_local', 'rr_local_std', 'rr_local_cv']

def rr_features(samples):
    samples = np.asarray(samples, dtype=np.int64)
    if len(samples) < 2 or (np.diff(samples) <= 0).any():
        raise ValueError('Need at least two strictly ordered beat positions')
    intervals = np.diff(samples) / FS
    pre, post = np.r_[np.nan, intervals], np.r_[intervals, np.nan]
    series = pd.Series(pre)
    local = series.rolling(10, center=True, min_periods=1).mean().to_numpy()
    local_std = series.rolling(10, center=True, min_periods=2).std(ddof=0).to_numpy()
    global_mean = np.full(len(samples), intervals.mean())
    return np.column_stack([pre, post, global_mean, local, pre/local, post/local,
                            pre/global_mean, post/global_mean, pre/post,
                            (post-pre)/local, local_std, local_std/local])

def extract_features(waves, rr):
    raw, normalized = waves[:, 0], waves[:, 1]
    # Fixed-interval morphology follows the paper's FP-centered QRS/T sampling idea.
    morphology_offsets = np.r_[np.linspace(-.05, .10, 10), np.linspace(.15, .50, 8)] * FS + LEFT
    full_offsets = np.linspace(0, WAVE_LENGTH-1, 64)
    morphology = np.vstack([np.interp(morphology_offsets, np.arange(WAVE_LENGTH), w) for w in raw])
    morphology_norm = np.vstack([np.interp(morphology_offsets, np.arange(WAVE_LENGTH), w) for w in normalized])
    shape = np.vstack([np.interp(full_offsets, np.arange(WAVE_LENGTH), w) for w in normalized])
    delta = np.diff(raw, axis=1)
    stats = np.column_stack([raw.mean(1), raw.std(1), raw.min(1), raw.max(1), np.median(raw, axis=1),
        np.ptp(raw, axis=1), np.sqrt((raw**2).mean(1)), (raw**2).sum(1),
        skew(raw, axis=1), kurtosis(raw, axis=1), delta.std(1), delta.min(1), delta.max(1)])
    packets = []
    for w in raw:
        nodes = pywt.WaveletPacket(w, 'db4', mode='symmetric', maxlevel=4).get_level(4, order='freq')
        energy = np.array([np.square(node.data).sum() for node in nodes])
        relative = energy / max(energy.sum(), 1e-12)
        entropy = []
        for node in nodes:
            e = np.square(node.data); p = e / max(e.sum(), 1e-12)
            entropy.append(float(-np.sum(p * np.log(p + 1e-12))))
        packets.append(np.r_[relative, entropy])
    names = RR_NAMES + [n+'_missing' for n in RR_NAMES]
    names += [f'paper_morphology_{i:02d}' for i in range(18)]
    names += [f'normalized_morphology_{i:02d}' for i in range(18)]
    names += [f'normalized_shape_{i:02d}' for i in range(64)]
    names += ['mean','std','min','max','median','peak_to_peak','rms','energy','skew','kurtosis',
              'diff_std','diff_min','diff_max']
    names += [f'packet_relative_energy_{i:02d}' for i in range(16)]
    names += [f'packet_entropy_{i:02d}' for i in range(16)]
    values = np.column_stack([rr, ~np.isfinite(rr), morphology, morphology_norm, shape, stats, packets])
    values[~np.isfinite(values)] = np.nan
    return values.astype(np.float32), names

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride=1):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=False), nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Conv1d(out_channels, out_channels, kernel, padding=kernel // 2, bias=False), nn.BatchNorm1d(out_channels))
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return F.relu(self.main(x) + self.skip(x))

class ECGNetwork(nn.Module):

    def __init__(self, kind, parameters, context_dim, n_classes=N_CLASSES):
        super().__init__()
        layers, previous = ([], 2)
        for stage in range(parameters['stages']):
            width = min(parameters['base_channels'] * 2 ** stage, 256)
            for block in range(parameters['blocks_per_stage']):
                layers.append(ResidualBlock(previous, width, parameters['kernel'], stride=2 if block == 0 else 1))
                previous = width
        self.conv = nn.Sequential(*layers)
        self.kind = kind
        if kind == 'cnn_bilstm':
            hidden = parameters['lstm_hidden']
            self.lstm = nn.LSTM(previous, hidden, num_layers=parameters['lstm_layers'], batch_first=True, bidirectional=True, dropout=parameters['dropout'] if parameters['lstm_layers'] > 1 else 0.0)
            previous = 2 * hidden
        self.rr_branch = nn.Sequential(nn.Linear(context_dim, parameters['rr_hidden']), nn.LayerNorm(parameters['rr_hidden']), nn.ReLU())
        combined = 2 * previous + parameters['rr_hidden']
        self.head = nn.Sequential(nn.Linear(combined, parameters['head_hidden']), nn.LayerNorm(parameters['head_hidden']), nn.ReLU(), nn.Dropout(parameters['dropout']), nn.Linear(parameters['head_hidden'], n_classes))

    def forward(self, waveform, context):
        encoded = self.conv(waveform)
        if self.kind == 'cnn_bilstm':
            encoded, _ = self.lstm(encoded.transpose(1, 2))
            encoded = encoded.transpose(1, 2)
        pooled = torch.cat([encoded.mean(2), encoded.amax(2)], dim=1)
        return self.head(torch.cat([pooled, self.rr_branch(context)], dim=1))


def choose_lead(names):
    if 'MLII' not in names:
        raise ValueError('Benchmark requires MLII; other leads cannot be substituted')
    return names.index('MLII'), 'MLII'


def read_record(path):
    record = wfdb.rdrecord(str(path))
    if record.fs != FS:
        raise ValueError('Benchmark requires 360 Hz')
    index, lead = choose_lead(record.sig_name)
    signal = np.asarray(record.p_signal[:, index], dtype=np.float64)
    if len(signal) != record.sig_len or not np.isfinite(signal).all():
        raise ValueError('Incomplete or nonfinite ECG signal')
    return signal, lead


def read_beats(path):
    """Keep Q positions for RR context, even though Q is not a prediction target."""
    ann = wfdb.rdann(str(path), 'atr')
    mask = np.array([s in ALL_LABEL_MAP for s in ann.symbol], dtype=bool)
    return np.asarray(ann.sample)[mask], np.asarray(ann.symbol)[mask]


def segment(signal, samples):
    samples = np.asarray(samples)
    if (samples.ndim != 1 or not np.issubdtype(samples.dtype, np.integer)
            or len(samples) < 2 or np.any(np.diff(samples) <= 0)):
        raise ValueError('Need at least two strictly increasing integer beat positions')
    if np.any(samples < 0) or np.any(samples >= len(signal)):
        raise ValueError('Beat positions are outside the signal')
    filtered = filter_ecg(signal)
    valid = (samples >= LEFT) & (samples + RIGHT < len(signal))
    if not valid.any():
        raise ValueError('No complete 271-sample beat windows')
    windows = np.array([filtered[r-LEFT:r+RIGHT+1] for r in samples[valid]], dtype=np.float32)
    record_scale = max(float(filtered.std()), 1e-6)
    beat_std = np.maximum(windows.std(axis=1, keepdims=True), 1e-6)
    waves = np.stack([windows / record_scale,
                      (windows - windows.mean(axis=1, keepdims=True)) / beat_std], axis=1)
    return waves.astype(np.float32), valid


def validate_contract(config, state, schema):
    expected = {'protocol': SUPPORTED_PROTOCOL, 'classes': CLASSES,
                'left_samples': LEFT, 'right_samples': RIGHT, 'offline_context': True,
                'normalization': 'per_record_std_and_per_beat_zscore',
                'filter': 'median_200ms_600ms_then_zero_phase_lowpass_35Hz'}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported benchmark setting: {key}')
    if state.get('classes') != CLASSES or state.get('shape') != [2, WAVE_LENGTH] or state.get('offline') is not True:
        raise ValueError('Incompatible preprocessing shape/classes/context')
    _, names = extract_features(np.zeros((1, 2, WAVE_LENGTH), dtype=np.float32), np.ones((1, 12)))
    if (schema.get('names') != names or schema.get('rr_names') != RR_NAMES
            or state.get('rr_names') != RR_NAMES or schema.get('wavelet') != 'db4'
            or schema.get('wavelet_packet_level') != 4):
        raise ValueError('Incompatible feature schema')


def load_model(folder, info, device='cpu'):
    folder = Path(folder)
    kind = info['model']
    if info['classes'] != CLASSES:
        raise ValueError('Model class order differs from N/S/V/F')
    if kind == 'rf':
        return joblib.load(folder / 'model.joblib')
    if kind == 'xgb':
        model = xgb.Booster()
        model.load_model(folder / 'model.ubj')
        model.set_param({'device': 'cpu', 'nthread': 1})
        return model
    if kind not in ['cnn', 'cnn_bilstm']:
        raise ValueError(f'Unsupported model: {kind}')
    model = ECGNetwork(kind, info['parameters'], info['context_dim'])
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
    return model.to(device).eval()


def probabilities(model, kind, part, device='cpu', batch_size=128):
    if kind == 'rf':
        output = np.zeros((len(part['x']), N_CLASSES), dtype=np.float32)
        output[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(part['x'])
        return output
    if kind == 'xgb':
        return model.predict(xgb.DMatrix(part['x'], nthread=1))
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(part['wave']), batch_size):
            wave = torch.as_tensor(part['wave'][start:start+batch_size], device=device)
            context = torch.as_tensor(part['context'][start:start+batch_size], device=device)
            outputs.append(model(wave, context).softmax(1).cpu().numpy())
    return np.concatenate(outputs)


def load_bundle(bundle):
    bundle = Path(bundle)
    manifest = read_json(bundle / 'bundle.json')
    if manifest.get('version') != 2:
        raise ValueError('Export a four-class bundle (version 2) with notebook 10')
    kind = manifest.get('model')
    model_file = {'rf': 'model.joblib', 'xgb': 'model.ubj', 'cnn': 'model.pt', 'cnn_bilstm': 'model.pt'}.get(kind)
    required = {'info.json', 'transforms.joblib', 'benchmark_config.json', 'preprocessing.json',
                'feature_schema.json', 'evaluation.json', model_file}
    if model_file is None or not required.issubset(manifest.get('files', {})):
        raise ValueError('Incomplete bundle manifest')
    for name, checksum in manifest['files'].items():
        if Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('Invalid bundle filename')
        if sha256(bundle / name) != checksum:
            raise ValueError(f'Bundle file changed: {name}')
    config = read_json(bundle / 'benchmark_config.json')
    state = read_json(bundle / 'preprocessing.json')
    schema = read_json(bundle / 'feature_schema.json')
    validate_contract(config, state, schema)
    if state['config_hash'] != sha256(bundle / 'benchmark_config.json'):
        raise ValueError('Preprocessing/config mismatch')
    info = read_json(bundle / 'info.json')
    if info['model'] != kind or info['context_dim'] != 24 or manifest.get('classes') != CLASSES:
        raise ValueError('Model metadata mismatch')
    device = 'cuda' if kind in ['cnn', 'cnn_bilstm'] and torch.cuda.is_available() else 'cpu'
    model = load_model(bundle, info, device)
    transforms = joblib.load(bundle / 'transforms.joblib')
    if set(transforms) != {'feature_imputer', 'rr_imputer', 'rr_scaler'}:
        raise ValueError('Incompatible fitted transforms')
    return model, info, transforms, schema, device


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


STAGES = ['รับและตรวจข้อมูล', 'เตรียมโมเดล', 'ประมวลผลข้อมูล', 'เตรียม input โมเดล', 'จำแนก heartbeat', 'เตรียมผลลัพธ์']


def predict_record(bundle, record, beat_samples=None, progress=None):
    timings = {}

    def step(name, function, device='cpu'):
        if progress:
            progress(name, 'running', None)
        synchronize(device)
        started = perf_counter()
        try:
            value = function()
            synchronize(device)
        except Exception:
            if progress:
                progress(name, 'error', perf_counter() - started)
            raise
        elapsed = perf_counter() - started
        timings[name] = elapsed
        if progress:
            progress(name, 'complete', elapsed)
        return value

    def read_input():
        signal, lead = read_record(record)
        # Predictions cover all valid supplied beat positions. Q truth is outside the task,
        # but its position remains in RR context, exactly as during preparation.
        samples = read_beats(record)[0] if beat_samples is None else np.asarray(beat_samples)
        return signal, lead, samples

    signal, lead, samples = step(STAGES[0], read_input)
    model, info, transforms, schema, device = step(STAGES[1], lambda: load_bundle(bundle))
    waves, valid = step(STAGES[2], lambda: segment(signal, samples))

    def prepare_input():
        rr = rr_features(samples)[valid]
        values, names = extract_features(waves, rr)
        if names != schema['names']:
            raise ValueError('Feature order mismatch')
        return transform_partition({'wave': waves, 'rr': rr, 'x': values}, transforms)

    part = step(STAGES[3], prepare_input)
    proba = step(STAGES[4], lambda: probabilities(model, info['model'], part, device), device)

    def results():
        if proba.shape != (len(waves), N_CLASSES) or not np.isfinite(proba).all():
            raise ValueError('Invalid four-class model output')
        output = pd.DataFrame({'r_sample': samples[valid], 'predicted_id': proba.argmax(1)})
        output['predicted_class'] = [CLASSES[i] for i in output.predicted_id]
        for i, name in enumerate(CLASSES):
            output[f'score_{name}'] = proba[:, i]
        return output

    output = step(STAGES[5], results)
    return {'predictions': output, 'signal': signal, 'lead': lead, 'timings': timings,
            'processing_seconds': sum(timings.values()), 'device': device,
            'model_load': 'โหลดโมเดลใหม่ในรอบนี้', 'model': info['model'],
            'beats': len(output), 'fs': FS}
