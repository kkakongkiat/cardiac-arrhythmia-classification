from collections import Counter
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import wfdb
from sklearn.impute import SimpleImputer

from program.common import (RAW, DATA, CLASSES, LABEL_MAP, SEED, PREPROCESSING,
                            read_json, write_json, sha256, new_stage)
from program.preprocessing import read_record, segment, rr_features, features


def audit(raw=RAW):
    new_stage(DATA / 'audit.json')
    records_file = raw / 'RECORDS'
    if not records_file.exists():
        raise FileNotFoundError(f'วาง MIT-BIH พร้อม RECORDS ที่ {raw}')
    records = records_file.read_text().split()
    if len(records) != 48 or len(set(records)) != 48 or not all(r.isdigit() for r in records):
        raise ValueError('ต้องใช้ 48 records หลักจาก RECORDS เท่านั้น')
    DATA.mkdir(parents=True, exist_ok=True)
    rows, inventory, excluded, checksums = [], [], [], {}
    nonbeats = set('~+[]!"|x()p t u `\'sT=^@')
    for record_id in records:
        for ext in ['hea', 'dat', 'atr']:
            path = raw / f'{record_id}.{ext}'
            checksums[path.name] = sha256(path)
        signal, lead = read_record(raw / record_id)
        ann = wfdb.rdann(str(raw / record_id), 'atr')
        if np.any(ann.sample < 0) or np.any(ann.sample >= len(signal)):
            raise ValueError(f'{record_id}: annotation bounds ไม่ถูกต้อง')
        patient = '201_202' if record_id in ['201', '202'] else record_id
        counts = Counter(ann.symbol)
        for r, symbol in zip(ann.sample, ann.symbol):
            if symbol not in LABEL_MAP:
                excluded.append({'record_id': record_id, 'r_sample': int(r), 'symbol': symbol,
                                 'reason': 'non_beat' if symbol in nonbeats else 'unmapped_review'})
                continue
            rows.append({'beat_id': f'{record_id}:{r}', 'record_id': record_id,
                         'patient_group': patient, 'r_sample': int(r),
                         'original_symbol': symbol, 'class_id': LABEL_MAP[symbol], 'lead': lead})
        inventory.append({'record_id': record_id, 'patient_group': patient,
                          'lead': lead, 'samples': len(signal), 'symbol_counts': dict(counts)})
    beats = pd.DataFrame(rows)
    if beats.beat_id.duplicated().any():
        raise ValueError('beat_id ซ้ำ ต้องตรวจ annotations')
    beats.to_parquet(DATA / 'inventory.parquet', index=False)
    pd.DataFrame(excluded, columns=['record_id', 'r_sample', 'symbol', 'reason']).to_csv(
        DATA / 'excluded_annotations.csv', index=False)
    beats.groupby(['patient_group', 'class_id']).size().unstack(fill_value=0).to_csv(
        DATA / 'patient_class_counts.csv')
    result = {'dataset_version': 'mit-bih-arrhythmia-database-1.0.0', 'files': checksums,
              'records': inventory, 'excluded_symbols': dict(Counter(x['symbol'] for x in excluded)),
              'unmapped_symbols': sorted({x['symbol'] for x in excluded if x['reason'] == 'unmapped_review'})}
    write_json(DATA / 'audit.json', result)
    return beats


def validate_split(beats):
    if set(beats.split) != {'train', 'validation', 'test'}:
        raise ValueError('ต้องมีเพียง train/validation/test')
    if beats.groupby('patient_group').split.nunique().max() != 1:
        raise ValueError('พบ patient leakage')
    for name, part in beats.groupby('split'):
        if set(part.class_id) != set(range(5)):
            raise ValueError(f'{name} มีคลาสไม่ครบ 5; หยุดก่อนฝึก ห้ามแก้ split จากคะแนนโมเดล')
    paired = beats[beats.record_id.isin(['201', '202'])]
    if not paired.empty and (paired.patient_group.nunique() != 1 or paired.split.nunique() != 1):
        raise ValueError('201/202 ต้องเป็นผู้ป่วยกลุ่มเดียวและชุดเดียวกัน')


def create_split(candidates=20000):
    new_stage(DATA / 'split.json')
    audit_info = read_json(DATA / 'audit.json')
    if audit_info['unmapped_symbols']:
        raise ValueError(f"ตรวจ unmapped symbols ก่อน: {audit_info['unmapped_symbols']}")
    beats = pd.read_parquet(DATA / 'inventory.parquet')
    counts = pd.crosstab(beats.patient_group, beats.class_id).reindex(columns=range(5), fill_value=0)
    groups = counts.index.to_numpy()
    # 47 ผู้ป่วย -> 37/5/5 โดยไม่แยกคนเดียวกันข้ามชุด
    n_holdout = max(1, round(len(groups) * .1))
    rng = np.random.default_rng(SEED)
    total = counts.to_numpy().sum(axis=0)
    if (total == 0).any():
        raise ValueError('Dataset มีคลาสไม่ครบ')
    best = None
    for _ in range(candidates):
        order = rng.permutation(len(groups))
        parts = [order[2*n_holdout:], order[:n_holdout], order[n_holdout:2*n_holdout]]
        sizes = np.array([counts.iloc[p].sum().to_numpy() for p in parts])
        if (sizes == 0).any():
            continue
        # เลือกด้วย class coverage และสัดส่วนเท่านั้น ไม่อ่านคะแนนโมเดล
        score = float(np.abs(sizes / total - np.array([.8, .1, .1])[:, None]).mean())
        if best is None or score < best[0]:
            best = (score, parts)
    if best is None:
        raise ValueError('ไม่พบ split ครบ 5 คลาสในงบค้นหา; ตรวจจำนวนผู้ป่วยต่อคลาสก่อนแก้ protocol')
    mapping = {str(groups[i]): name for name, ids in zip(['train', 'validation', 'test'], best[1]) for i in ids}
    beats['split'] = beats.patient_group.map(mapping)
    validate_split(beats)
    manifest = {'seed': SEED, 'target': [0.8, 0.1, 0.1], 'patient_split': mapping,
                'audit_hash': sha256(DATA / 'audit.json'), 'search_candidates': candidates,
                'record_split': dict(zip(beats.record_id, beats.split))}
    manifest['split_hash'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    beats.to_parquet(DATA / 'split_inventory.parquet', index=False)
    write_json(DATA / 'split.json', manifest)
    summary = beats.groupby('split').agg(patients=('patient_group', 'nunique'), beats=('beat_id', 'size'))
    summary.join(pd.crosstab(beats.split, beats.class_id)).to_csv(DATA / 'split_summary.csv')
    return summary


def prepare(raw=RAW):
    new_stage(DATA / 'preprocessing.json')
    split = read_json(DATA / 'split.json')
    audit_info = read_json(DATA / 'audit.json')
    if sha256(DATA / 'audit.json') != split['audit_hash']:
        raise ValueError('audit เปลี่ยนหลัง split')
    for name, expected in audit_info['files'].items():
        if sha256(raw / name) != expected:
            raise ValueError(f'raw file เปลี่ยน: {name}')
    inventory = pd.read_parquet(DATA / 'split_inventory.parquet')
    chunks, waves, rrs, exclusions = [], [], [], []
    for record_id, part in inventory.groupby('record_id', sort=True):
        part = part.sort_values('r_sample').reset_index(drop=True)
        signal, _ = read_record(raw / record_id)
        samples = part.r_sample.to_numpy()
        wave, valid = segment(signal, samples)
        rr = rr_features(samples)[valid]
        exclusions.extend(part.loc[~valid, ['beat_id', 'record_id']].assign(reason='window_boundary').to_dict('records'))
        chunks.append(part.loc[valid].copy())
        waves.append(wave)
        rrs.append(rr)
    meta = pd.concat(chunks, ignore_index=True)
    wave = np.concatenate(waves)
    rr = np.concatenate(rrs)
    validate_split(meta)
    # Fit เฉพาะ Train; validation/test ได้ transform ด้วยค่าเดียวกัน
    train_wave = wave[meta.split.eq('train').to_numpy()]
    mean, std = float(train_wave.mean(dtype=np.float64)), float(train_wave.std(dtype=np.float64))
    if std <= 1e-12:
        raise ValueError('Train waveform ไม่มี variance')
    state = {**PREPROCESSING, 'mean': mean, 'std': std, 'split_hash': split['split_hash'],
             'audit_hash': split['audit_hash'], 'classes': CLASSES, 'label_map': LABEL_MAP}
    for name in ['train', 'validation', 'test']:
        mask = meta.split.eq(name).to_numpy()
        folder = DATA / name
        folder.mkdir(parents=True, exist_ok=True)
        selected = meta.loc[mask].copy().reset_index(drop=True)
        selected['waveform_index'] = np.arange(len(selected))
        selected.to_parquet(folder / 'metadata.parquet', index=False)
        np.save(folder / 'waveforms.npy', ((wave[mask] - mean) / std).astype(np.float32))
        np.save(folder / 'rr.npy', rr[mask])
    pd.DataFrame(exclusions, columns=['beat_id', 'record_id', 'reason']).to_csv(DATA / 'excluded_beats.csv', index=False)
    write_json(DATA / 'preprocessing.json', state)
    return meta.groupby(['split', 'class_id']).size().unstack(fill_value=0)


def prepare_features():
    new_stage(DATA / 'prepared.json')
    state = read_json(DATA / 'preprocessing.json')
    frames = {}
    for split in ['train', 'validation', 'test']:
        folder = DATA / split
        frames[split] = features(np.load(folder / 'waveforms.npy'), np.load(folder / 'rr.npy'))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    imputer.fit(frames['train'])
    joblib.dump(imputer, DATA / 'imputer.joblib')
    state['feature_order'] = list(frames['train'].columns)
    write_json(DATA / 'preprocessing.json', state)
    for split, frame in frames.items():
        folder = DATA / split
        frame.to_parquet(folder / 'features.parquet', index=False)
        np.save(folder / 'features.npy', imputer.transform(frame).astype(np.float32))
    # แยก artifact ของ Test ไว้ต่างหาก โค้ดฝึกไม่โหลดไฟล์เหล่านี้
    files = [p for p in DATA.rglob('*') if p.is_file() and p.name != 'prepared.json']
    write_json(DATA / 'prepared.json', {'split_hash': state['split_hash'],
               'files': {p.relative_to(DATA).as_posix(): sha256(p) for p in files}})
    return {'feature_count': len(state['feature_order']), 'split_hash': state['split_hash']}


def load_partition(split, *, allow_test=False):
    if split not in ['train', 'validation', 'test']:
        raise ValueError('Unknown partition')
    if split == 'test' and not allow_test:
        raise PermissionError('Test เปิดได้เฉพาะ final evaluation หลังล็อกโมเดล')
    manifest = read_json(DATA / 'prepared.json')
    folder = DATA / split
    for name in ['metadata.parquet', 'waveforms.npy', 'features.npy']:
        if sha256(folder / name) != manifest['files'][f'{split}/{name}']:
            raise ValueError(f'artifact เปลี่ยน: {split}/{name}')
    meta = pd.read_parquet(folder / 'metadata.parquet')
    waves, x = np.load(folder / 'waveforms.npy'), np.load(folder / 'features.npy')
    if len(meta) != len(waves) or len(meta) != len(x) or meta.beat_id.duplicated().any():
        raise ValueError('beat alignment ไม่ถูกต้อง')
    if not meta.split.eq(split).all() or set(meta.class_id) != set(range(5)):
        raise ValueError('partition หรือ class coverage ไม่ถูกต้อง')
    if not np.array_equal(meta.waveform_index.to_numpy(), np.arange(len(meta))):
        raise ValueError('waveform index ไม่ตรงกับ metadata')
    if waves.shape != (len(meta), 252) or not np.isfinite(waves).all() or not np.isfinite(x).all():
        raise ValueError('ขนาด waveform หรือค่าข้อมูลหลัง preprocessing ไม่ถูกต้อง')
    return {'meta': meta, 'wave': waves, 'x': x, 'y': meta.class_id.to_numpy()}
