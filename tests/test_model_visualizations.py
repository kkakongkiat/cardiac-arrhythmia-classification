"""Check real plotted values and training histories with small synthetic data."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from tests.notebook_support import notebook_definitions


def test_forest_curve_final_point_matches_saved_forest_predictions():
    ns = notebook_definitions('model/6_Random_forest.ipynb')
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 8)).astype(np.float32)
    y = np.tile(np.arange(5), 10)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(x, y)
    partition = {'x': x, 'y': y}
    curve = ns['forest_learning_curve'](model, partition, partition, points=3, train_limit=20)
    assert curve.n_estimators.tolist() == [1, 3, 5]
    assert curve.iloc[-1].validation_macro_f1 == pytest.approx(ns['macro_f1'](y, model.predict(x)))
    assert curve[['train_macro_f1', 'validation_macro_f1']].notna().all().all()


@pytest.mark.parametrize('notebook', ['8_CNN', '9_CNN_BiLSTM'])
def test_validation_loss_matches_full_weighted_cross_entropy(notebook):
    ns = notebook_definitions(f'model/{notebook}.ipynb')
    rng = np.random.default_rng(42)
    values = rng.normal(size=(10, 252)).astype(np.float32)
    labels = np.tile(np.arange(5), 2)
    model = ns['network'](ns['MODEL'], {'channels': [2], 'hidden': 2, 'kernel': 3, 'dropout': 0.}).eval()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 3., 4., 5.]))
    with torch.inference_mode():
        logits = model(torch.from_numpy(values[:, None, :]))
        expected = criterion(logits, torch.from_numpy(labels)).item()
    predicted, loss = ns['validation_epoch_metrics'](model, values, labels, criterion, 'cpu', 3)
    assert loss == pytest.approx(expected, rel=1e-6)
    np.testing.assert_array_equal(predicted, logits.argmax(1).numpy())


def test_dl_training_records_both_losses_and_scores(tmp_path):
    ns = notebook_definitions('model/8_CNN.ipynb')
    rng = np.random.default_rng(42)
    partition = {'wave': rng.normal(size=(10, 252)).astype(np.float32),
                 'y': np.tile(np.arange(5), 2)}
    params = {'channels': [2], 'kernel': 3, 'dropout': 0., 'lr': .001, 'weight_decay': 0.}
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        ns['_fit_network']('cnn', params, partition, partition, None, tmp_path, 'cpu', 5, 2)
    finally:
        torch.set_num_threads(threads)
    history = pd.read_csv(tmp_path / 'history.csv')
    assert history.epoch.tolist() == [1, 2]
    assert np.isfinite(history[['train_loss', 'validation_loss', 'train_macro_f1',
                                'validation_macro_f1']].to_numpy()).all()


@pytest.mark.parametrize('legacy', [False, True])
def test_dl_learning_plot_supports_old_and_new_histories(tmp_path, legacy):
    ns = notebook_definitions('model/8_CNN.ipynb')
    folder, output = tmp_path / 'cnn_unweighted/baseline', tmp_path / 'plots'
    folder.mkdir(parents=True)
    output.mkdir()
    history = {'epoch': [1, 2], 'train_loss': [.8, .5], 'validation_macro_f1': [.3, .4]}
    if not legacy:
        history.update(validation_loss=[.9, .7], train_macro_f1=[.4, .6])
    pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
    fig = ns['plot_training_curve'](folder, {'best_epoch_or_round': 2}, output)
    expected = history['train_loss']
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), expected)
    score_line = next(line for line in fig.axes[1].lines if line.get_label().startswith('Validation'))
    np.testing.assert_array_equal(score_line.get_ydata(), history['validation_macro_f1'])
    assert (output / 'learning_curve.png').stat().st_size > 1000
    plt.close(fig)


def test_xgb_curve_uses_saved_rounds(tmp_path):
    ns = notebook_definitions('model/7_XGBoost.ipynb')
    folder, output = tmp_path / 'xgb_unweighted/trial_00', tmp_path / 'plots'
    folder.mkdir(parents=True)
    output.mkdir()
    history = {'train': {'macro_f1': [.4, .6, .8]}, 'validation': {'macro_f1': [.3, .5, .4]}}
    (folder / 'history.json').write_text(json.dumps(history))
    fig = ns['plot_training_curve'](folder, {'best_epoch_or_round': 2}, output)
    np.testing.assert_array_equal(fig.axes[0].lines[1].get_ydata(), history['validation']['macro_f1'])
    np.testing.assert_array_equal(fig.axes[0].lines[2].get_xdata(), [2, 2])
    plt.close(fig)


def test_tuning_progress_keeps_baseline_when_trials_worse(tmp_path):
    ns = notebook_definitions('model/6_Random_forest.ipynb')
    pd.DataFrame({'number': [0, 1, 2], 'value': [.3, None, .5],
                  'state': ['COMPLETE', 'FAIL', 'COMPLETE']}).to_csv(tmp_path / 'trials.csv', index=False)
    selected = {'baseline_macro_f1': .6, 'configuration': 'rf_unweighted', 'candidate': 'baseline'}
    fig = ns['plot_tuning_progress'](tmp_path, selected, tmp_path)
    line = next(line for line in fig.axes[0].lines if line.get_label() == 'Best including baseline')
    np.testing.assert_array_equal(line.get_ydata(), [.6, .6])
    np.testing.assert_array_equal(line.get_xdata(), [1, 3])
    plt.close(fig)
