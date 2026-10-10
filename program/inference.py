from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import torch

from program.common import CLASSES, PREPROCESSING, read_json, sha256
from program.preprocessing import read_record, beat_samples as read_beats, segment, rr_features, features
from program.models import load_model, probabilities, synchronize

STAGES = ['รับและตรวจข้อมูล', 'เตรียมโมเดล', 'ประมวลผลข้อมูล',
          'เตรียม input โมเดล', 'จำแนก heartbeat', 'เตรียมผลลัพธ์']


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
