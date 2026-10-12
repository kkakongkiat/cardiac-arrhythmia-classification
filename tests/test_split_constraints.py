"""Patient split constraints, using synthetic counts without training models."""
import numpy as np
import pandas as pd
import pytest

from tests.notebook_support import notebook_definitions


definitions = notebook_definitions('dataset/2_Data_split.ipynb')
find_patient_split = definitions['find_patient_split']
validate_split = definitions['validate_split']
NAMES = ['train', 'validation', 'test']
QUOTAS = dict.fromkeys(NAMES, 2)
MINIMA = {name: {'N': 1, 'S': 1, 'V': 1, 'F': 15, 'Q': 10} for name in NAMES}


def patient_counts():
    return pd.DataFrame(
        [[100, 10, 20, 10, q] for q in [20, 0, 20, 0, 20, 0]],
        index=['100', '101', '103', '105', '106', '107'], columns=range(5),
    )


def metadata(counts, mapping):
    return pd.DataFrame([
        {'patient_group': group, 'record_id': group, 'class_id': cls, 'split': mapping[group]}
        for group, row in counts.iterrows()
        for cls, count in row.items()
        for _ in range(int(count))
    ])


def test_constrained_split_preserves_groups_and_minimum_support():
    counts = patient_counts()
    mapping, info = find_patient_split(counts, [.5, .25, .25], QUOTAS, MINIMA, time_limit=5)
    frame = metadata(counts, mapping)
    validate_split(frame, min_class_counts=MINIMA, expected_patients=QUOTAS)
    assert set(mapping) == set(counts.index)
    assert frame.groupby('patient_group').split.nunique().eq(1).all()
    actual = pd.crosstab(frame.split, frame.class_id).loc[NAMES].to_numpy()
    expected_error = np.abs(actual / counts.sum().to_numpy() - np.array([.5, .25, .25])[:, None]).mean()
    assert info['allocation_error'] == pytest.approx(expected_error)


def test_minimum_support_rejects_nonempty_but_tiny_class():
    counts = patient_counts()
    mapping = dict(zip(counts.index, ['train', 'train', 'validation', 'validation', 'test', 'test']))
    frame = metadata(counts, mapping)
    q_test = frame.index[frame.split.eq('test') & frame.class_id.eq(4)]
    frame = frame.drop(q_test[1:])
    with pytest.raises(ValueError, match='minimum class support for Q'):
        validate_split(frame, min_class_counts=MINIMA, expected_patients=QUOTAS)


def test_infeasible_patient_distribution_is_not_relaxed():
    counts = patient_counts()
    counts.loc['106', 4] = 0  # Enough Q beats in total, but only two Q-positive patients.
    with pytest.raises(ValueError, match='No feasible patient split'):
        find_patient_split(counts, [.5, .25, .25], QUOTAS, MINIMA, time_limit=5)


def test_insufficient_total_support_is_rejected():
    impossible = {name: {**minimum, 'Q': 100} for name, minimum in MINIMA.items()}
    with pytest.raises(ValueError, match='insufficient total beats'):
        find_patient_split(patient_counts(), [.5, .25, .25], QUOTAS, impossible, time_limit=5)
