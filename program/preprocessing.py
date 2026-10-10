import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample
from scipy.stats import skew, kurtosis
import wfdb

from program.common import LABEL_MAP, PREPROCESSING


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


def beat_samples(path):
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
