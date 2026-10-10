"""Synthetic save/load checks for the user to run later; no MIT-BIH data needed."""
import joblib
import numpy as np
import pytest
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from program.models import network, load_model, probabilities


@pytest.mark.parametrize('kind', ['rf', 'xgb', 'cnn', 'cnn_bilstm'])
def test_model_save_load_predictions_match(tmp_path, kind):
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    x = rng.normal(size=(20, 8)).astype(np.float32)
    y = np.tile(np.arange(5), 4)
    parameters = {'channels': [4, 8], 'kernel': 5, 'hidden': 4, 'dropout': .1}
    if kind == 'rf':
        model = RandomForestClassifier(n_estimators=3, random_state=42).fit(x, y)
        joblib.dump(model, tmp_path / 'model.joblib')
    elif kind == 'xgb':
        model = xgb.train({'objective': 'multi:softprob', 'num_class': 5,
                           'max_depth': 2, 'seed': 42, 'nthread': 1},
                          xgb.DMatrix(x, label=y), num_boost_round=2)
        model.save_model(tmp_path / 'model.ubj')
    else:
        x = rng.normal(size=(3, 252)).astype(np.float32)
        model = network(kind, parameters).eval()
        # serialization test ของน้ำหนัก ไม่ใช่ training/accuracy test
        torch.save(model.state_dict(), tmp_path / 'model.pt')
    expected = probabilities(model, kind, x)
    restored = load_model(tmp_path, {'kind': kind, 'parameters': parameters})
    actual = probabilities(restored, kind, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.shape == (len(x), 5)
