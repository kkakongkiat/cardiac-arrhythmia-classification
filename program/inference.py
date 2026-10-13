"""App inference only; training and evaluation live directly in notebooks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample
from scipy.stats import skew, kurtosis
import wfdb
import joblib
import torch
from torch import nn
import xgboost as xgb
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset/raw/mit-bih-arrhythmia-database-1.0.0'
ARTIFACTS = ROOT / 'model/artifacts'
CLASSES = ['N', 'S', 'V', 'F', 'Q']
SYMBOLS = ['NLRej', 'AaJS', 'VE', 'F', '/fQ']
LABEL_MAP = {s: i for i, group in enumerate(SYMBOLS) for s in group}
PREPROCESSING = {'version': 1, 'fs': 360, 'left': 90, 'right': 162,
                 'band': [0.5, 40], 'order': 4, 'leads': ['MLII', 'V5']}

def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'ไม่พบ {path}; กรุณารัน notebook ขั้นก่อนหน้าก่อน')
    return json.loads(path.read_text(encoding='utf-8'))

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def choose_lead(names):
    for name in PREPROCESSING['leads']:
        if name in names:
            return names.index(name), name
    raise ValueError(f'ไม่มี lead MLII หรือ V5: {names}')

def read_record(path):
    record = wfdb.rdrecord(str(path))
    if record.fs != PREPROCESSING['fs']:
        raise ValueError(f'ต้องการ 360 Hz แต่พบ {record.fs}')
    index, lead = choose_lead(record.sig_name)
    signal = np.asarray(record.p_signal[:, index], dtype=np.float64)
    if len(signal) != record.sig_len or not np.isfinite(signal).all():
        # ไม่ filter ข้าม NaN เพราะจะกระจายความเสียหายไปทั้ง record
        raise ValueError(f'{path}: signal ไม่ครบหรือมี NaN/Inf ต้องตรวจสอบก่อน')
    return signal, lead

def read_beats(path):
    annotation = wfdb.rdann(str(path), 'atr')
    # ใช้ชนิด annotation เพื่อแยก beat จาก rhythm events; ไม่ใช้ class เป็น feature
    mask = np.array([symbol in LABEL_MAP for symbol in annotation.symbol])
    return np.asarray(annotation.sample)[mask], np.asarray(annotation.symbol)[mask]

def segment(signal, samples):
    samples = np.asarray(samples, dtype=np.int64)
    if samples.ndim != 1 or np.any(np.diff(samples) <= 0):
        raise ValueError('ตำแหน่ง heartbeat ต้องเรียงและไม่ซ้ำ')
    if np.any(samples < 0) or np.any(samples >= len(signal)):
        raise ValueError('ตำแหน่ง annotation อยู่นอก signal')
    sos = butter(4, [0.5, 40], btype='bandpass', fs=360, output='sos')
    filtered = sosfiltfilt(sos, signal)
    valid = (samples >= 90) & (samples + 162 <= len(signal))
    waves = np.array([filtered[r - 90:r + 162] for r in samples[valid]], dtype=np.float32)
    if not len(waves):
        raise ValueError('ไม่มี heartbeat ที่ตัดหน้าต่าง 252 samples ได้')
    if waves.shape[1] != 252 or not np.isfinite(waves).all():
        raise ValueError('waveform หลัง filter ใช้งานไม่ได้')
    return waves, valid

def rr_features(samples):
    # คำนวณบนลำดับ beats ทั้ง record ก่อนตัด boundary beats ทิ้ง
    intervals = np.diff(samples) / 360.0
    previous = np.r_[np.nan, intervals]
    following = np.r_[intervals, np.nan]
    trailing = pd.Series(previous).rolling(10, min_periods=1).mean().to_numpy()
    return np.column_stack([previous, following, previous / following, trailing])

def features(waves, rr):
    delta = np.diff(waves, axis=1)
    stats = np.column_stack([
        waves.mean(1), waves.std(1), waves.min(1), waves.max(1),
        np.median(waves, axis=1), np.ptp(waves, axis=1),
        np.sqrt(np.mean(waves ** 2, axis=1)), np.sum(waves ** 2, axis=1),
        skew(waves, axis=1), kurtosis(waves, axis=1),
        delta.mean(1), delta.std(1), delta.min(1), delta.max(1)])
    names = ['rr_previous', 'rr_next', 'rr_ratio', 'rr_trailing_10',
             'mean', 'std', 'min', 'max', 'median', 'peak_to_peak', 'rms', 'energy',
             'skew', 'kurtosis', 'diff_mean', 'diff_std', 'diff_min', 'diff_max']
    names += [f'morphology_{i:02d}' for i in range(64)]
    values = np.column_stack([rr, stats, resample(waves, 64, axis=1)])
    values[~np.isfinite(values)] = np.nan
    frame = pd.DataFrame(values, columns=names)
    # indicator คงที่ทุกคอลัมน์ ป้องกัน schema เปลี่ยนเมื่อ missing ใน inference
    return pd.concat([frame, frame.isna().astype(float).add_suffix('_missing')], axis=1)

class ECGNetwork(nn.Module):
    def __init__(self, kind, channels, kernel=7, hidden=64, dropout=.3):
        super().__init__()
        layers, previous = [], 1
        for width in channels:
            layers += [nn.Conv1d(previous, width, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)]
            previous = width
        self.conv = nn.Sequential(*layers)
        self.kind = kind
        if kind == 'cnn_bilstm':
            self.lstm = nn.LSTM(previous, hidden, batch_first=True, bidirectional=True)
            previous = hidden * 2
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(previous, 5))

    def forward(self, x):
        x = self.conv(x)
        if self.kind == 'cnn_bilstm':
            _, (hidden, _) = self.lstm(x.transpose(1, 2))
            x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            x = x.mean(dim=2)
        return self.head(x)

def network(kind, parameters):
    return ECGNetwork(kind, parameters['channels'], parameters.get('kernel', 7),
                      parameters.get('hidden', 64), parameters.get('dropout', .3))

def load_model(folder, info, device='cpu'):
    kind = info['kind']
    if kind == 'rf':
        return joblib.load(folder / 'model.joblib')
    if kind == 'xgb':
        model = xgb.Booster()
        model.load_model(folder / 'model.ubj')
        return model
    model = network(kind, info['parameters'])
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
    return model.to(device).eval()

def probabilities(model, kind, values, device='cpu', batch_size=128):
    if kind == 'rf':
        output = np.zeros((len(values), 5), dtype=np.float32)
        output[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(values)
        return output
    if kind == 'xgb':
        return model.predict(xgb.DMatrix(values))
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            # waveform: (beats, 252) -> Conv1d: (beats, 1, 252)
            batch = torch.as_tensor(values[start:start + batch_size, None, :], device=device)
            outputs.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)

def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()

STAGES = ['รับและตรวจข้อมูล', 'เตรียมโมเดล', 'ประมวลผลข้อมูล', 'เตรียม input โมเดล', 'จำแนก heartbeat', 'เตรียมผลลัพธ์']

def predict_record(bundle, record, beat_samples=None, progress=None):
    bundle, record = Path(bundle), Path(record)
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
        # callback/UI รวมถึงเวลาหน่วงเรียกหลังหยุดจับเวลาขั้นประมวลผลแล้ว
        if progress:
            progress(name, 'complete', elapsed)
        return value

    def read_input():
        signal, lead = read_record(record)
        samples = read_beats(record)[0] if beat_samples is None else np.asarray(beat_samples)
        if not len(samples):
            raise ValueError('ไม่พบ heartbeat annotation ที่รองรับ')
        return signal, lead, samples

    signal, lead, samples = step(STAGES[0], read_input)

    def open_bundle():
        manifest = read_json(bundle / 'bundle.json')
        if manifest['version'] != 1:
            raise ValueError('ไม่รองรับ bundle version นี้')
        for name, checksum in manifest['files'].items():
            if sha256(bundle / name) != checksum:
                raise ValueError(f'bundle file เปลี่ยน: {name}')
        state = read_json(bundle / 'preprocessing.json')
        for name, expected in PREPROCESSING.items():
            if state[name] != expected:
                raise ValueError(f'Preprocessing ไม่เข้ากัน: {name}')
        if state['classes'] != CLASSES:
            raise ValueError('Class order ไม่ตรงกัน')
        info = read_json(bundle / 'info.json')
        device = 'cuda' if info['kind'] in ['cnn', 'cnn_bilstm'] and torch.cuda.is_available() else 'cpu'
        model = load_model(bundle, info, device)
        imputer = joblib.load(bundle / 'imputer.joblib')
        synchronize(device)
        return model, info, state, imputer, device

    model, info, state, imputer, device = step(STAGES[1], open_bundle)

    def process():
        wave, valid = segment(signal, samples)
        return ((wave - state['mean']) / state['std']).astype(np.float32), valid

    waves, valid = step(STAGES[2], process)

    def prepare_input():
        if info['kind'] in ['cnn', 'cnn_bilstm']:
            return waves
        frame = features(waves, rr_features(samples)[valid])
        if list(frame.columns) != state['feature_order']:
            raise ValueError('Feature order ไม่ตรงกับตอนฝึก')
        return imputer.transform(frame).astype(np.float32)

    values = step(STAGES[3], prepare_input)
    proba = step(STAGES[4], lambda: probabilities(model, info['kind'], values, device), device)

    def results():
        output = pd.DataFrame({'r_sample': samples[valid], 'predicted_id': proba.argmax(1)})
        output['predicted_class'] = [CLASSES[i] for i in output.predicted_id]
        for i, name in enumerate(CLASSES):
            output[f'score_{name}'] = proba[:, i]
        return output

    output = step(STAGES[5], results)
    return {'predictions': output, 'signal': signal, 'lead': lead, 'timings': timings,
            'processing_seconds': sum(timings.values()), 'device': device,
            'model_load': 'โหลดโมเดลใหม่ในรอบนี้', 'beats': len(output), 'fs': state['fs']}
