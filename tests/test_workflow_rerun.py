"""Exercise evaluation/export reruns with temporary synthetic artifacts, never MIT-BIH."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

from tests.notebook_support import ROOT, notebook_definitions


def section(relative, heading):
    cells = json.loads((ROOT / relative).read_text(encoding='utf-8'))['cells']
    index = next(i for i, c in enumerate(cells)
                 if c['cell_type'] == 'markdown' and ''.join(c['source']).strip() == heading)
    return ''.join(cells[index + 1]['source'])


def set_output_roots(namespace, tmp_path):
    for name, folder in [('DATA', 'data'), ('ARTIFACTS', 'artifacts'), ('REPORTS', 'reports')]:
        namespace[name] = tmp_path / folder
        namespace[name].mkdir(exist_ok=True)


@pytest.mark.parametrize('notebook', ['6_Random_forest', '7_XGBoost',
                                     '8_CNN', '9_CNN_BiLSTM',
                                     '11_Evaluation_comparison', '12_Export_model'])
def test_replace_output_cleans_only_requested_child(tmp_path, notebook):
    namespace = notebook_definitions(f'model/{notebook}.ipynb')
    set_output_roots(namespace, tmp_path)
    child = namespace['ARTIFACTS'] / 'candidate'
    child.mkdir()
    (child / 'old.pt').write_text('old checkpoint')
    sibling = namespace['ARTIFACTS'] / 'keep.txt'
    sibling.write_text('other model')
    reset = namespace['reset_directory']
    for _ in range(2):
        reset(child)
        assert list(child.iterdir()) == []
        assert sibling.read_text() == 'other model'
    for invalid in [tmp_path, namespace['ARTIFACTS'], tmp_path / 'raw']:
        with pytest.raises(ValueError):
            reset(invalid)
    assert sibling.exists()


@pytest.mark.parametrize('notebook', ['6_Random_forest', '7_XGBoost', '8_CNN', '9_CNN_BiLSTM'])
def test_tuning_uses_editable_parameter_cell(notebook):
    namespace = notebook_definitions(f'model/{notebook}.ipynb')
    class Trial:
        def suggest_categorical(self, name, choices):
            return choices[-1]

        def suggest_float(self, name, low, high, log=False):
            return high

    namespace['TUNING_SPACE'] = {'edited_parameter': {'type': 'float', 'low': .1, 'high': .42}}
    if namespace['MODEL'] in ['cnn', 'cnn_bilstm']:
        namespace['TUNING_SPACE']['width'] = {'type': 'categorical', 'choices': ['custom']}
        namespace['CHANNEL_OPTIONS'] = {'custom': [3, 6]}
    result = namespace['search_parameters'](Trial(), namespace['MODEL'])
    assert result['edited_parameter'] == .42
    if namespace['MODEL'] in ['cnn', 'cnn_bilstm']:
        assert result['channels'] == [3, 6]


def test_evaluate_select_export_and_rerun(tmp_path):
    relative = 'model/11_Evaluation_comparison.ipynb'
    namespace = notebook_definitions(relative)
    set_output_roots(namespace, tmp_path)
    data, artifacts, reports = (namespace[key] for key in ['DATA', 'ARTIFACTS', 'REPORTS'])
    write, digest = namespace['write_json'], namespace['sha256']
    y = np.tile(np.arange(5), 2)
    meta = pd.DataFrame({'beat_id': [f'beat_{i}' for i in range(10)], 'class_id': y,
                         'waveform_index': np.arange(10), 'split': 'test',
                         'patient_group': ['p1'] * 5 + ['p2'] * 5,
                         'record_id': ['100'] * 5 + ['101'] * 5, 'lead': 'MLII'})
    test = data / 'test'
    test.mkdir()
    meta.to_parquet(test / 'metadata.parquet', index=False)
    np.save(test / 'waveforms.npy', np.zeros((10, 252), dtype=np.float32))
    np.save(test / 'features.npy', np.zeros((10, 8), dtype=np.float32))
    for name in ['preprocessing.json', 'split.json', 'imputer.joblib']:
        (data / name).write_text('synthetic preprocessing')
    write(data / 'prepared.json', {'files': {p.relative_to(data).as_posix(): digest(p)
                                           for p in data.rglob('*') if p.is_file()}})
    names = [f'{kind}_{weight}' for kind in ['rf', 'xgb', 'cnn', 'cnn_bilstm']
             for weight in ['unweighted', 'weighted']]
    for name in names:
        kind = name.rsplit('_', 1)[0]
        folder = artifacts / name / 'baseline'
        folder.mkdir(parents=True)
        extension = 'pt' if kind in ['cnn', 'cnn_bilstm'] else 'joblib' if kind == 'rf' else 'ubj'
        (folder / f'model.{extension}').write_text(name)
        prepared_hash = digest(data / 'prepared.json')
        write(folder / 'info.json', {'kind': kind, 'weighted': name.endswith('_weighted'),
                                    'device': 'cpu', 'prepared_hash': prepared_hash, 'train_seconds': 1.})
        write(folder.parent / 'selected.json', {'configuration': name, 'candidate': 'baseline',
                                               'prepared_hash': prepared_hash, 'baseline_macro_f1': .8,
                                               'validation_macro_f1': .99 if kind == 'rf' else .8,
                                               'tuning_seconds': 0.})
    # Only inference/latency are substituted. The notebook reads/checks files, computes metrics,
    # writes real reports, selects the winner and exports through its actual workflow cells.
    best = ['cnn_unweighted']
    namespace['load_model'] = lambda folder, info, device: folder.parent.name
    def probabilities(model, kind, values, device):
        prediction = y if model == best[0] else np.zeros_like(y)
        return np.eye(5, dtype=np.float32)[prediction]
    namespace['probabilities'] = probabilities
    namespace['measure_latency'] = lambda *args, **kwargs: {
        'batch_1_seconds': .01, 'batch_1_actual_count': 1,
        'batch_128_seconds': .02, 'batch_128_actual_count': 10}
    namespace['LATENCY_REPEATS'] = 1
    namespace['BOOTSTRAP_REPEATS'] = 10
    export_relative = 'model/12_Export_model.ipynb'
    export = notebook_definitions(export_relative)
    for key in ['DATA', 'ARTIFACTS', 'REPORTS']:
        export[key] = namespace[key]
    export_workflow = [c for c in json.loads((ROOT / export_relative).read_text(encoding='utf-8'))['cells']
                       if c['cell_type'] == 'code'][-2:]
    for expected in ['cnn_unweighted', 'rf_weighted']:
        best[0] = expected
        for heading in ['## Load Test and replace previous reports', '## Evaluate and inspect errors',
                        '## Select best Test Macro-F1', '## Save comparison and limitations',
                        '## Visualize best model and compare all configurations']:
            exec(section(relative, heading), namespace)
        selection = json.loads((artifacts / 'selection.json').read_text())
        assert selection['winner'] == expected
        assert selection['test_used_for_selection'] is True
        assert selection['selection_metric'] == 'test_macro_f1'
        table = pd.read_csv(reports / 'final/comparison.csv')
        assert len(table) == 8
        assert table.loc[table.selected_winner, 'configuration'].tolist() == [expected]
        assert not (reports / 'final/old_report.txt').exists()
        assert len(list((reports / 'final').glob('*/patient_bootstrap.json'))) == 1
        for name in ['winner_confusion_matrix', 'model_score_comparison', 'per_class_f1_comparison',
                     'inference_time_comparison', 'score_vs_inference_time']:
            assert (reports / f'final/{name}.png').stat().st_size > 1000
        near_best = pd.read_csv(reports / 'final/near_best_inference.csv')
        assert near_best.configuration.tolist() == [expected]
        assert near_best.iloc[0].batch_128_ms_per_beat == pytest.approx(2.)
        for code in export_workflow:
            exec(''.join(code['source']), export)
        bundle = json.loads((artifacts / 'exported/bundle.json').read_text())
        assert bundle['winner'] == expected
        assert bundle['selection_metric'] == 'test_macro_f1'
        assert bundle['test_used_for_selection'] is True
        assert not (artifacts / 'exported/old.bin').exists()
        assert len(list((artifacts / 'exported').glob('model.*'))) == 1
        for name, expected_hash in bundle['files'].items():
            assert digest(artifacts / 'exported' / name) == expected_hash
        (reports / 'final/old_report.txt').write_text('previous report')
        (artifacts / 'exported/old.bin').write_text('previous export')
    # Changing a checkpoint after evaluation requires reevaluation before export.
    (artifacts / 'rf_weighted/baseline/model.joblib').write_text('changed')
    with pytest.raises(ValueError, match='Checkpoint/artifact'):
        export['verify_selection']()
