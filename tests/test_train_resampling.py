"""Independent split undersampling must preserve non-N beats and feature alignment."""
import numpy as np
import pandas as pd

from tests.notebook_support import notebook_definitions


def test_split_n_resampling_preserves_other_classes_and_rr_alignment():
    ns = notebook_definitions('dataset/3_Preprocessing.ipynb')
    rows = []
    expected = {'train': [40, 4, 8, 2, 5], 'validation': [20, 2, 3, 1, 6],
                'test': [15, 3, 4, 1, 2]}
    for split, counts in expected.items():
        for cls, count in enumerate(counts):
            for _ in range(count):
                rows.append({'beat_id': str(len(rows)), 'record_id': split,
                             'patient_group': split, 'split': split, 'class_id': cls})
    meta = pd.DataFrame(rows)
    wave = np.arange(len(meta))[:, None].repeat(252, axis=1)
    rr = np.arange(len(meta))[:, None].repeat(4, axis=1) + 0.5
    config = {'enabled': True, 'method': 'random_undersampling',
              'class_label': 'N', 'splits': ['train', 'validation', 'test'],
              'n_to_largest_other_ratio': 1.0, 'seed': 42}
    sample = ns['undersample_n_by_split']
    selected, waves, rrs, removed, summary = sample(meta, wave, rr, config)
    for split, counts in expected.items():
        assert summary['splits'][split]['counts_after'] == dict(zip(
            ['N', 'S', 'V', 'F', 'Q'], [max(counts[1:]), *counts[1:]],
        ))
    assert len(removed) == 57
    assert removed.class_id.eq(0).all()
    retained = meta.loc[meta.class_id.ne(0)]
    pd.testing.assert_frame_equal(
        selected.loc[selected.class_id.ne(0)].reset_index(drop=True),
        retained.reset_index(drop=True),
    )
    positions = selected.beat_id.astype(int).to_numpy()
    np.testing.assert_array_equal(waves, wave[positions])
    np.testing.assert_array_equal(rrs, rr[positions])
    pd.testing.assert_frame_equal(selected, sample(meta, wave, rr, config)[0])
    # Run the notebook's actual application and normalization cells together:
    # normalization must not restore the original unsampled arrays.
    import json
    from tests.notebook_support import ROOT
    book = json.loads((ROOT / 'dataset/3_Preprocessing.ipynb').read_text(encoding='utf-8'))
    ns.update(chunks=[meta], waves=[wave], rrs=[rr], audit_info={'resampling': config},
              split={'split_hash': 'synthetic', 'audit_hash': 'synthetic'}, display=lambda _: None)
    for ident in ['resampling-apply', 'cell-020']:
        source = next(c['source'] for c in book['cells'] if c.get('id') == ident)
        exec(''.join(source), ns)
    assert len(ns['meta']) == len(selected)
    np.testing.assert_array_equal(ns['rr'], rrs)
    assert ns['state']['resampling']['removed_n_beats'] == 57
    # Sampling can be disabled, or request more N than available without duplicating beats.
    for change in [{'enabled': False}, {'n_to_largest_other_ratio': 10.0}]:
        unchanged, _, _, dropped, _ = sample(meta, wave, rr, {**config, **change})
        pd.testing.assert_frame_equal(unchanged, meta)
        assert dropped.empty

    # Turning off sampling in one split must not change draws in the others.
    train_only, _, _, _, _ = sample(meta, wave, rr, {**config, 'splits': ['train']})
    pd.testing.assert_frame_equal(
        train_only.loc[train_only.split.eq('train')].reset_index(drop=True),
        selected.loc[selected.split.eq('train')].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        train_only.loc[train_only.split.ne('train')].reset_index(drop=True),
        meta.loc[meta.split.ne('train')].reset_index(drop=True),
    )
