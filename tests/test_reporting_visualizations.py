"""Exercise notebook reporting without loading ECG data or running tuning."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def cells(name):
    return json.loads((ROOT / 'model' / name).read_text(encoding='utf-8'))['cells']


def source(cell):
    return ''.join(cell['source'])


def load_function(name, function, namespace):
    nodes = [node for cell in cells(name) if cell['cell_type'] == 'code'
             for node in ast.parse(source(cell)).body
             if isinstance(node, ast.FunctionDef) and node.name == function]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, name, 'exec'), namespace)
    return namespace[function]


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.ns = {'np': np, 'pd': pd, 'plt': plt, 'CLASSES': ['N', 'S', 'V', 'F'],
                   'N_CLASSES': 4, 'display': lambda value: None}

    def tearDown(self):
        plt.close('all')

    def test_confusion_orientation_and_true_class_normalization(self):
        plot = load_function('9_Evaluation_comparison.ipynb', 'plot_confusion_matrix', self.ns)
        counts = np.array([[7, 2, 1, 0], [3, 1, 0, 0], [0, 1, 5, 0], [0, 0, 0, 0]])
        for normalized in [False, True]:
            fig = plot({'confusion_matrix': counts}, 'Example', normalized=normalized)
            ax = fig.axes[0]
            matrix = np.asarray(ax.images[0].get_array())
            self.assertEqual(ax.get_xlabel(), 'True')
            self.assertEqual(ax.get_ylabel(), 'Predicted')
            if normalized:
                np.testing.assert_allclose(matrix.sum(axis=0), [1, 1, 1, 0])
                self.assertAlmostEqual(matrix[1, 0], 0.2)
            else:
                np.testing.assert_array_equal(matrix, counts.T)
                self.assertEqual(matrix[1, 0], 2)

    def test_forest_curve_matches_saved_forest_predictions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 5))
        y = np.tile(np.arange(4), 20)
        train = {'x': x[:60], 'y': y[:60]}
        validation = {'x': x[60:], 'y': y[60:]}
        model = RandomForestClassifier(n_estimators=12, max_depth=4, random_state=42).fit(train['x'], train['y'])
        self.ns['macro_f1'] = lambda actual, predicted: f1_score(actual, predicted, labels=range(4), average='macro', zero_division=0)
        curve_fn = load_function('5_Random_forest.ipynb', 'forest_learning_curve', self.ns)
        predictions_before = model.predict_proba(validation['x']).copy()
        curve = curve_fn(model, train, validation, checkpoints=5)
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve.iloc[-1]['n_estimators'], 12)
        for name, part in [('train', train), ('validation', validation)]:
            expected = self.ns['macro_f1'](part['y'], model.predict(part['x']))
            self.assertAlmostEqual(curve.iloc[-1][name + '_macro_f1'], expected)
        np.testing.assert_array_equal(model.predict_proba(validation['x']), predictions_before)

    def test_recorded_histories_and_saved_parameter_summaries(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(plt, 'show'):
            destination = Path(folder)
            self.ns['destination'] = destination
            for filename, kind in [('6_XGBoost.ipynb', 'xgb'), ('7_CNN.ipynb', 'cnn'), ('8_CNN_BiLSTM.ipynb', 'cnn_bilstm')]:
                history = ({'validation': {'mlogloss': [1.2, 0.9, 0.8], 'macro_f1': [0.2, 0.4, 0.35]}}
                           if kind == 'xgb' else [{'epoch': i, 'train_loss': 1 / i, 'validation_macro_f1': 0.1 * i} for i in range(1, 4)])
                self.ns.update(MODEL=kind, selected_info={'trial': 2, 'best_steps': 2}, read_json=lambda path: history)
                exec(source(cells(filename)[21]), self.ns)
                self.assertTrue((destination / 'selected_history.png').is_file())
                self.assertEqual(self.ns['axes'][0].get_xlabel(), 'Boosting round' if kind == 'xgb' else 'Epoch')
            selected = {'parameters': {'max_depth': None, 'n_estimators': 12}, 'best_steps': 12,
                        'trial': 2, 'seed': 42, 'device': 'cpu', 'class_weights': [1, 2, 3, 4],
                        'validation_macro_f1': 0.5}
            final = {**selected, 'fit_partition': 'train+validation', 'selected_trial': 2, 'training_beats': 80}
            self.ns.update(MODEL='rf', read_json=lambda path: selected if path.parent.name == 'validation_best' else final)
            exec(source(cells('5_Random_forest.ipynb')[-1]), self.ns)
            self.assertEqual(self.ns['hyperparameter_summary'].loc['n_estimators', 'final_model'], '12')
            self.assertEqual(self.ns['fit_summary'].loc['Fit partition', 'final_model'], 'train+validation')

    def test_model_comparison_bars(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(plt, 'show'):
            reports = Path(folder)
            (reports / 'final').mkdir()
            keys = ['accuracy', 'precision_macro', 'recall_macro', 'macro_f1', 'weighted_f1', 'balanced_accuracy']
            keys += [label + '_f1-score' for label in self.ns['CLASSES']]
            comparison = pd.DataFrame([{'name': name, **{key: 0.7 + i * 0.05 for key in keys}}
                                       for i, name in enumerate(['Random Forest', 'XGBoost', 'CNN + RR', 'CNN-BiLSTM + RR'])])
            self.ns.update(comparison=comparison, REPORTS=reports)
            exec(source(cells('9_Evaluation_comparison.ipynb')[11]), self.ns)
            self.assertEqual(len(self.ns['axes'][0].patches), 24)
            self.assertEqual(len(self.ns['axes'][1].patches), 16)
            self.assertTrue((reports / 'final' / 'comparison.png').is_file())


if __name__ == '__main__':
    unittest.main()
