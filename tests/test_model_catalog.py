"""All-model export, provenance checks and ranked desktop selection."""
import json
import shutil
import threading

import pandas as pd
import pytest

from program.inference import sha256
from program.service import discover_models, analyze, CLASS_DETAILS
from tests.test_benchmark_application import record, make_bundle, export_function, write_json
from tests.test_desktop_application import ui, tk_root, sample_result


@pytest.fixture
def evaluated_project(tmp_path, record):
    project = tmp_path / 'all'
    snapshots = {}
    order = ['xgb', 'rf', 'cnn_bilstm', 'cnn']
    for kind in order:
        single = tmp_path / f'single-{kind}'
        _, _, final = make_bundle(single, kind, record)
        destination = project / 'model/artifacts' / kind / 'example/final'
        shutil.copytree(final, destination)
        snapshots[kind] = {'path': f'{kind}/example/final', 'files': {
            p.name: sha256(p) for p in destination.iterdir()}}
    shutil.copytree(single / 'dataset', project / 'dataset')
    shutil.copy2(single / 'benchmark_config.json', project / 'benchmark_config.json')
    report = project / 'model/reports/final'
    report.mkdir(parents=True)
    pd.DataFrame({'model': order, 'macro_f1': [.9, .8, .7, .6]}).to_csv(report / 'comparison.csv', index=False)
    evaluation = json.loads((single / 'model/reports/final/evaluation.json').read_text())
    evaluation.update(benchmark_top_model='xgb', artifact_snapshots=snapshots,
                      comparison_hash=sha256(report / 'comparison.csv'))
    write_json(report / 'evaluation.json', evaluation)
    return project


def test_all_models_export_load_and_selection(evaluated_project, record, ui):
    folder, catalog = export_function('export_all_models')(evaluated_project)
    assert [entry['model'] for entry in catalog['models']] == ['xgb', 'rf', 'cnn_bilstm', 'cnn']
    entries = discover_models(str(folder))
    assert [entry['score'] for entry in entries] == [.9, .8, .7, .6]
    for entry in entries:
        result = analyze(record, entry['bundle'], lambda *a: None, threading.Event())
        assert result['model'] == entry['model']
        assert result['predictions'].shape[0] == 4
    ui.bundle.set(str(folder))
    assert ui.selected_bundle() == entries[0]['bundle']
    ui.accept_result(sample_result())
    ui.model_choice.set(entries[2]['label'])
    ui.model_box.event_generate('<<ComboboxSelected>>')
    ui.root.update()
    assert ui.result is None
    assert ui.selected_bundle() == entries[2]['bundle']
    ui.accept_result(sample_result())
    ui.beat.set('3')
    ui.select_beat()
    text = ui.explanation.get('1.0', 'end')
    assert 'ข้อเสนอแนะ' in text and 'ventricular ectopic' in text


def test_changed_nonwinner_preserves_export(evaluated_project):
    folder, _ = export_function('export_all_models')(evaluated_project)
    before = sha256(folder / 'catalog.json')
    checkpoint = evaluated_project / 'model/artifacts/cnn/example/final/model.pt'
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Evaluated artifact changed'):
        export_function('export_all_models')(evaluated_project)
    assert sha256(folder / 'catalog.json') == before
    assert len(discover_models(str(folder))) == 4


def test_catalog_rejects_score_tampering(evaluated_project):
    folder, catalog = export_function('export_all_models')(evaluated_project)
    catalog['models'][0]['test_macro_f1'] = .1
    write_json(folder / 'catalog.json', catalog)
    with pytest.raises(ValueError):
        discover_models(str(folder))


def test_advice_available_for_every_class():
    assert set(CLASS_DETAILS) == set('NSVF')
    for _, _, explanation in CLASS_DETAILS.values():
        assert 'ผลเบื้องต้น' in explanation and 'ข้อเสนอแนะ' in explanation
