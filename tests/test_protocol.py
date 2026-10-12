"""Prepared tests only; not executed as part of file delivery."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from program.inference import LABEL_MAP, choose_lead, segment
from tests.notebook_support import notebook_definitions

split_definitions = notebook_definitions('dataset/2_Data_split.ipynb')
training_definitions = notebook_definitions('model/6_Random_forest.ipynb')
evaluation_definitions = notebook_definitions('model/11_Evaluation_comparison.ipynb')
validate_split = split_definitions['validate_split']
load_partition = training_definitions['load_partition']
macro_f1 = training_definitions['macro_f1']
weights_from_train = training_definitions['weights_from_train']
verify_lock = evaluation_definitions['verify_lock']


def sample_metadata():
    rows = []
    for split, record in [('train', '100'), ('validation', '101'), ('test', '103')]:
        for cls in range(5):
            rows.append({'split': split, 'patient_group': record, 'record_id': record, 'class_id': cls})
    return pd.DataFrame(rows)


def test_patient_overlap_rejected():
    frame = sample_metadata()
    validate_split(frame)
    frame.loc[frame.split == 'test', 'patient_group'] = '100'
    with pytest.raises(ValueError, match='leakage'):
        validate_split(frame)


def test_paired_records_cannot_cross_split():
    frame = sample_metadata()
    frame.loc[frame.split == 'train', 'record_id'] = '201'
    frame.loc[frame.split == 'test', 'record_id'] = '202'
    with pytest.raises(ValueError, match='201/202'):
        validate_split(frame)


def test_missing_class_rejected():
    frame = sample_metadata()
    frame = frame[~((frame.split == 'validation') & (frame.class_id == 4))]
    with pytest.raises(ValueError, match='คลาสไม่ครบ'):
        validate_split(frame)


def test_test_partition_denied_before_read():
    # ไม่ต้องมีไฟล์ dataset ก็ต้องปฏิเสธตั้งแต่ guard
    with pytest.raises(PermissionError):
        load_partition('test')


def test_evaluation_requires_lock(tmp_path, monkeypatch):
    monkeypatch.setitem(evaluation_definitions, 'ARTIFACTS', tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_lock()


def test_labels_leads_and_boundary():
    assert LABEL_MAP['/'] == LABEL_MAP['f'] == LABEL_MAP['Q'] == 4
    assert '+' not in LABEL_MAP
    assert choose_lead(['V5', 'MLII']) == (1, 'MLII')
    assert choose_lead(['V5', 'V2']) == (0, 'V5')
    with pytest.raises(ValueError):
        choose_lead(['V1'])
    signal = np.sin(np.arange(1000) / 20.)
    waves, valid = segment(signal, [20, 400, 950])
    assert waves.shape == (1, 252)
    assert valid.tolist() == [False, True, False]


def test_fixed_five_class_metric_and_weights():
    assert macro_f1(np.arange(5), np.zeros(5, dtype=int)) == pytest.approx(1 / 15)
    y = np.array([0, 0, 0, 1, 2, 3, 4])
    weights = weights_from_train(y)
    assert weights[1] == pytest.approx(3 * weights[0])


def test_delivered_notebooks_are_unexecuted():
    root = Path(__file__).resolve().parents[1]
    notebooks = list((root / 'dataset').glob('*.ipynb')) + list((root / 'model').glob('*.ipynb'))
    assert len(notebooks) == 12
    for path in notebooks:
        for cell in json.loads(path.read_text(encoding='utf-8'))['cells']:
            if cell['cell_type'] == 'code':
                assert cell['execution_count'] is None
                assert cell['outputs'] == []
