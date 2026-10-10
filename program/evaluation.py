from pathlib import Path
from time import perf_counter
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch

from program.common import ARTIFACTS, DATA, REPORTS, CLASSES, SEED, read_json, write_json, sha256, new_stage
from program.data import load_partition
from program.models import load_model, probabilities, synchronize
from program.training import macro_f1


def measure_latency(model, kind, values, device, repeats=20):
    result = {}
    for size in [1, 128]:
        batch = values[:size]
        for _ in range(3):
            probabilities(model, kind, batch, device, batch_size=size)
        timings = []
        for _ in range(repeats):
            synchronize(device)
            start = perf_counter()
            probabilities(model, kind, batch, device, batch_size=size)
            synchronize(device)
            timings.append(perf_counter() - start)
        result[f'batch_{size}_seconds'] = float(np.median(timings))
        result[f'batch_{size}_actual_count'] = len(batch)
    return result


def select_models():
    new_stage(ARTIFACTS / 'selection.json')
    manifest = read_json(DATA / 'prepared.json')
    # ตรวจ state และ imputer ด้วย เพื่อให้ model bundle ใช้ preprocessing ที่ฝึกจริง
    for name in ['preprocessing.json', 'imputer.joblib', 'split.json']:
        if sha256(DATA / name) != manifest['files'][name]:
            raise ValueError(f'Preprocessing เปลี่ยน: {name}')
    validation = load_partition('validation')
    names = [f'{kind}_{weight}' for kind in ['rf', 'xgb', 'cnn', 'cnn_bilstm']
             for weight in ['unweighted', 'weighted']]
    records = []
    for name in names + ['dummy_unweighted']:
        chosen = read_json(ARTIFACTS / name / 'selected.json')
        path = ARTIFACTS / name / chosen['candidate']
        info = read_json(path / 'info.json')
        if info['prepared_hash'] != sha256(DATA / 'prepared.json'):
            raise ValueError(f'{name} ใช้ข้อมูลคนละ version')
        device = info['device']
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('ต้องวัด latency บน GPU เดียวกับ training')
        model = load_model(path, info, device)
        values = validation['wave'] if info['kind'] in ['cnn', 'cnn_bilstm'] else validation['x']
        latency = measure_latency(model, info['kind'], values, device)
        hashes = {p.name: sha256(p) for p in path.iterdir() if p.is_file()}
        records.append({**chosen, 'path': path.relative_to(ARTIFACTS).as_posix(),
                        'kind': info['kind'], 'weighted': info['weighted'], 'device': device,
                        'train_seconds': info['train_seconds'], 'latency': latency, 'files': hashes})
        del model
    scores = records[:8]
    best = max(r['validation_macro_f1'] for r in scores)
    eligible = [r for r in scores if best - r['validation_macro_f1'] <= .005]
    winner = min(eligible, key=lambda r: (r['latency']['batch_1_seconds'], r['configuration']))
    lock = {'winner': winner['configuration'], 'selection_metric': 'validation_macro_f1',
            'tie_tolerance': .005, 'prepared_hash': sha256(DATA / 'prepared.json'),
            'preprocessing_hash': sha256(DATA / 'preprocessing.json'),
            'imputer_hash': sha256(DATA / 'imputer.joblib'), 'candidates': records,
            'seed': SEED, 'test_used_for_selection': False}
    write_json(ARTIFACTS / 'selection.json', lock)
    return pd.DataFrame([{k: r[k] for k in ['configuration', 'candidate', 'validation_macro_f1']} for r in records])


def verify_lock():
    lock = read_json(ARTIFACTS / 'selection.json')
    if lock['prepared_hash'] != sha256(DATA / 'prepared.json'):
        raise ValueError('Dataset manifest เปลี่ยนหลังล็อกโมเดล')
    for filename, field in [('preprocessing.json', 'preprocessing_hash'), ('imputer.joblib', 'imputer_hash')]:
        if sha256(DATA / filename) != lock[field]:
            raise ValueError('Preprocessing เปลี่ยนหลังล็อกโมเดล')
    for candidate in lock['candidates']:
        for filename, expected in candidate['files'].items():
            if sha256(ARTIFACTS / candidate['path'] / filename) != expected:
                raise ValueError(f'Checkpoint/artifact เปลี่ยนหลังล็อก: {filename}')
    return lock


def metrics(y, pred):
    report = classification_report(y, pred, labels=list(range(5)), target_names=CLASSES,
                                   zero_division=0, output_dict=True)
    matrix = confusion_matrix(y, pred, labels=list(range(5)))
    normalized = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix, dtype=float),
                           where=matrix.sum(1, keepdims=True) != 0)
    return {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': macro_f1(y, pred),
            'balanced_accuracy': float(normalized.diagonal().mean()),
            'weighted_f1': float(report['weighted avg']['f1-score']), 'per_class': report,
            'confusion_matrix': matrix.tolist(), 'normalized_confusion_matrix': normalized.tolist()}


def bootstrap_patients(frame, repeats=1000):
    rng = np.random.default_rng(SEED)
    groups = {group: part for group, part in frame.groupby('patient_group')}
    names = list(groups)
    scores, absent = [], 0
    for _ in range(repeats):
        # สุ่มทั้งผู้ป่วยและคง beats ทั้งกลุ่ม รวมกลุ่มซ้ำตาม bootstrap multiplicity
        sample = pd.concat([groups[name] for name in rng.choice(names, len(names), replace=True)])
        absent += sample.class_id.nunique() < 5
        scores.append(macro_f1(sample.class_id, sample.predicted))
    return {'lower_95': float(np.quantile(scores, .025)), 'upper_95': float(np.quantile(scores, .975)),
            'replicates': repeats, 'missing_class_fraction': absent / repeats,
            'fixed_labels': CLASSES, 'seed': SEED}


def evaluate_locked():
    lock = verify_lock()  # ต้องผ่านขั้นนี้ก่อนเปิด Test
    target = REPORTS / 'final'
    target.mkdir(parents=True, exist_ok=False)
    test = load_partition('test', allow_test=True)
    rows = []
    for candidate in lock['candidates']:
        folder = ARTIFACTS / candidate['path']
        info = read_json(folder / 'info.json')
        model = load_model(folder, info, candidate['device'])
        values = test['wave'] if info['kind'] in ['cnn', 'cnn_bilstm'] else test['x']
        probabilities_ = probabilities(model, info['kind'], values, candidate['device'])
        pred = probabilities_.argmax(1)
        result = metrics(test['y'], pred)
        out = target / candidate['configuration']
        out.mkdir()
        frame = test['meta'].copy()
        frame['predicted'] = pred
        for i, name in enumerate(CLASSES):
            frame[f'score_{name}'] = probabilities_[:, i]
        frame.to_parquet(out / 'predictions.parquet', index=False)
        write_json(out / 'metrics.json', result)
        group_rows = []
        for column in ['record_id', 'patient_group', 'lead']:
            for group, subset in frame.groupby(column):
                group_rows.append({'group_type': column, 'group': str(group), 'beats': len(subset),
                                   'classes_present': int(subset.class_id.nunique()),
                                   'macro_f1_fixed_5': macro_f1(subset.class_id, subset.predicted)})
        pd.DataFrame(group_rows).to_csv(out / 'group_metrics.csv', index=False)
        errors = frame.loc[frame.class_id != frame.predicted]
        errors.to_csv(out / 'errors.csv', index=False)
        for key in ['confusion_matrix', 'normalized_confusion_matrix']:
            fig, ax = plt.subplots(figsize=(6, 5))
            plot = ax.imshow(result[key], cmap='Blues')
            ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel='Predicted', ylabel='True', title=candidate['configuration'])
            fig.colorbar(plot, ax=ax)
            fig.tight_layout()
            fig.savefig(out / f'{key}.png', dpi=160)
            plt.close(fig)
        # ตัวอย่างความผิดพลาดเลือกโดย deterministic ไม่เลือกเฉพาะภาพที่ดูดี
        if not errors.empty:
            example = errors.groupby(['class_id', 'predicted']).head(1).head(8)
            fig, axes = plt.subplots(len(example), 1, figsize=(9, 2 * len(example)), squeeze=False)
            for ax, (_, beat) in zip(axes[:, 0], example.iterrows()):
                ax.plot(test['wave'][int(beat.waveform_index)])
                ax.set_title(f"{beat.beat_id}: {CLASSES[int(beat.class_id)]} -> {CLASSES[int(beat.predicted)]}")
            fig.tight_layout()
            fig.savefig(out / 'error_examples.png', dpi=120)
            plt.close(fig)
        if candidate['configuration'] == lock['winner']:
            write_json(out / 'patient_bootstrap.json', bootstrap_patients(frame))
        row = {'configuration': candidate['configuration'], 'selected_winner': candidate['configuration'] == lock['winner'],
               'validation_macro_f1': candidate['validation_macro_f1'], 'baseline_validation_macro_f1': candidate['baseline_macro_f1'],
               'test_macro_f1': result['macro_f1'], 'accuracy': result['accuracy'],
               'balanced_accuracy': result['balanced_accuracy'], 'weighted_f1': result['weighted_f1'],
               'train_seconds': candidate['train_seconds'], 'tuning_seconds': candidate['tuning_seconds'],
               'model_bytes': sum(p.stat().st_size for p in folder.glob('model.*')), **candidate['latency']}
        for name in CLASSES:
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                row[f'{name}_{metric}'] = result['per_class'][name][metric]
        rows.append(row)
        del model
    comparison = pd.DataFrame(rows)
    comparison.to_csv(target / 'comparison.csv', index=False)
    write_json(target / 'evaluation.json', {'selection_hash': sha256(ARTIFACTS / 'selection.json'),
               'winner': lock['winner'], 'test_used_for_selection': False})
    (target / 'limitations.md').write_text(
        '# ข้อจำกัด\n\n- ใช้ seed เดียวและผู้ป่วย Test จำนวนน้อย\n'
        '- คะแนนกลุ่มย่อยใช้ 5 labels คงที่ ต้องอ่านจำนวนคลาสที่มีจริงประกอบ\n'
        '- ML ใช้ RR และ morphology; DL ใช้ waveform\n'
        '- ใช้ annotation centers และ offline filtering ไม่ได้ประเมิน R-peak detector\n'
        '- ผู้ชนะเลือกจาก Validation เท่านั้น ไม่เปลี่ยนตาม Test\n', encoding='utf-8')
    return comparison


def export_model():
    lock = verify_lock()
    evaluation = read_json(REPORTS / 'final/evaluation.json')
    if evaluation['selection_hash'] != sha256(ARTIFACTS / 'selection.json'):
        raise ValueError('ผล evaluate ไม่ตรงกับโมเดลที่ล็อก')
    winner = next(c for c in lock['candidates'] if c['configuration'] == lock['winner'])
    output = ARTIFACTS / 'exported'
    output.mkdir(parents=True, exist_ok=False)
    source = ARTIFACTS / winner['path']
    for path in source.glob('model.*'):
        shutil.copy2(path, output / path.name)
    shutil.copy2(source / 'info.json', output / 'info.json')
    for filename in ['preprocessing.json', 'imputer.joblib']:
        shutil.copy2(DATA / filename, output / filename)
    write_json(output / 'bundle.json', {'version': 1, 'winner': lock['winner'],
               'selection_hash': sha256(ARTIFACTS / 'selection.json'),
               'files': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    return output
