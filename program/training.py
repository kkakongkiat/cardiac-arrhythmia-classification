import gc
import random
from time import perf_counter
import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

from program.common import (ARTIFACTS, DATA, SEED, read_json, write_json, sha256,
                            environment, new_stage)
from program.data import load_partition
from program.models import network, probabilities, synchronize

DEFAULTS = {
    'dummy': {},
    'rf': {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
    'xgb': {'max_depth': 6, 'eta': .05, 'min_child_weight': 1, 'subsample': 1.,
            'colsample_bytree': 1., 'lambda': 1.},
    'cnn': {'channels': [32, 64, 128], 'kernel': 7, 'dropout': .3, 'lr': .001, 'weight_decay': .0001},
    'cnn_bilstm': {'channels': [32, 64], 'hidden': 64, 'kernel': 7, 'dropout': .3,
                    'lr': .001, 'weight_decay': .0001}}


def macro_f1(y, predicted):
    return float(f1_score(y, predicted, labels=list(range(5)), average='macro', zero_division=0))


def weights_from_train(y):
    counts = np.bincount(y, minlength=5)
    if (counts == 0).any():
        raise ValueError('Train ต้องมีครบ 5 คลาส')
    return len(y) / (5 * counts)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def search_parameters(trial, kind):
    if kind == 'rf':
        return {'n_estimators': trial.suggest_categorical('n_estimators', [300, 600]),
                'max_depth': trial.suggest_categorical('max_depth', [None, 15, 30]),
                'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 5]),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', .5])}
    if kind == 'xgb':
        return {'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7]),
                'eta': trial.suggest_float('eta', .01, .1, log=True),
                'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 5, 10]),
                'subsample': trial.suggest_float('subsample', .7, 1.),
                'colsample_bytree': trial.suggest_float('colsample_bytree', .7, 1.),
                'lambda': trial.suggest_float('lambda', .1, 10, log=True)}
    width = trial.suggest_categorical('width', ['small', 'large'])
    channels = ([16, 32, 64] if width == 'small' else [32, 64, 128]) if kind == 'cnn' else (
                [16, 32] if width == 'small' else [32, 64])
    values = {'channels': channels, 'kernel': 7,
              'dropout': trial.suggest_float('dropout', .1, .5),
              'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
              'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)}
    if kind == 'cnn':
        values['kernel'] = trial.suggest_categorical('kernel', [5, 7, 11])
    else:
        values['hidden'] = trial.suggest_categorical('hidden', [32, 64])
    return values


def _fit_network(kind, parameters, train, validation, weights, folder, device, batch_size, epochs):
    set_seed()
    model = network(kind, parameters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else torch.tensor(weights, dtype=torch.float32, device=device))
    ds = TensorDataset(torch.from_numpy(train['wave'][:, None, :]), torch.tensor(train['y'], dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED))
    best, stale, best_epoch = -1., 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        prediction = probabilities(model, kind, validation['wave'], device, batch_size).argmax(1)
        score = macro_f1(validation['y'], prediction)
        history.append({'epoch': epoch, 'train_loss': loss_sum / len(ds), 'validation_macro_f1': score})
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            # เก็บ epoch ที่ดีที่สุด ไม่ใช้ epoch สุดท้ายโดยอัตโนมัติ
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / 'model.pt')
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(torch.load(folder / 'model.pt', map_location=device, weights_only=True))
    pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
    return model, best_epoch


def fit_candidate(kind, weighted, parameters, folder, train, validation, *, epochs=60, rounds=1000):
    folder.mkdir(parents=True, exist_ok=False)
    set_seed()
    is_dl = kind in ['cnn', 'cnn_bilstm']
    device = 'cuda' if is_dl and torch.cuda.is_available() else 'cpu'
    if is_dl and device != 'cuda':
        raise RuntimeError('แผนนี้ใช้ GPU สำหรับ DL: ติดตั้ง PyTorch CUDA ก่อนรัน')
    weights = weights_from_train(train['y']) if weighted else None
    started = perf_counter()
    history, best_round, used_batch, oom_batches = {}, None, None, []
    if kind == 'dummy':
        model = DummyClassifier(strategy='most_frequent').fit(train['x'], train['y'])
        joblib.dump(model, folder / 'model.joblib')
    elif kind == 'rf':
        model = RandomForestClassifier(**parameters, random_state=SEED, n_jobs=-1,
                    class_weight=None if weights is None else dict(enumerate(weights)))
        model.fit(train['x'], train['y'])
        joblib.dump(model, folder / 'model.joblib')
    elif kind == 'xgb':
        dtrain = xgb.DMatrix(train['x'], label=train['y'], weight=None if weights is None else weights[train['y']])
        dval = xgb.DMatrix(validation['x'], label=validation['y'])
        def metric(pred, data):
            return 'macro_f1', macro_f1(data.get_label().astype(int), pred.argmax(axis=1))
        model = xgb.train({**parameters, 'objective': 'multi:softprob', 'num_class': 5,
                           'tree_method': 'hist', 'device': 'cpu', 'seed': SEED,
                           'disable_default_eval_metric': 1},
                          dtrain, num_boost_round=rounds, evals=[(dval, 'validation')],
                          custom_metric=metric, maximize=True, early_stopping_rounds=50,
                          evals_result=history, verbose_eval=False)
        best_round = model.best_iteration + 1
        model = model[:best_round]
        model.save_model(folder / 'model.ubj')
        write_json(folder / 'history.json', history)
    else:
        for batch_size in [128, 64, 32]:
            try:
                model, best_round = _fit_network(kind, parameters, train, validation, weights,
                                                  folder, device, batch_size, epochs)
                used_batch = batch_size
                break
            except torch.cuda.OutOfMemoryError:
                oom_batches.append(batch_size)
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size == 32:
                    raise
    synchronize(device)
    train_seconds = perf_counter() - started
    values = validation['wave'] if is_dl else validation['x']
    proba = probabilities(model, kind, values, device)
    score = macro_f1(validation['y'], proba.argmax(1))
    result = validation['meta'].copy()
    result['predicted'] = proba.argmax(1)
    for i in range(5):
        result[f'score_{i}'] = proba[:, i]
    result.to_parquet(folder / 'validation_predictions.parquet', index=False)
    info = {'kind': kind, 'weighted': weighted, 'seed': SEED, 'parameters': parameters,
            'validation_macro_f1': score, 'train_seconds': train_seconds, 'best_epoch_or_round': best_round,
            'batch_size': used_batch, 'oom_batch_attempts': oom_batches, 'device': device,
            'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'class_weights': None if weights is None else weights.tolist(),
            'prepared_hash': sha256(DATA / 'prepared.json'), 'environment': environment()}
    write_json(folder / 'info.json', info)
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return info


def run_configuration(kind, weighted=False, trials=12):
    name = f'{kind}_{"weighted" if weighted else "unweighted"}'
    folder = ARTIFACTS / name
    new_stage(folder / 'selected.json')
    # บังคับ fresh folder เมื่อ run ก่อนหน้าค้าง เพื่อไม่ใช้ trials ข้าม experiment
    if folder.exists():
        raise FileExistsError(f'{folder} มีผลบางส่วนอยู่แล้ว กรุณาตรวจและเก็บสำรองก่อนเริ่มใหม่')
    train, validation = load_partition('train'), load_partition('validation')
    folder.mkdir(parents=True)
    baseline = fit_candidate(kind, weighted, DEFAULTS[kind], folder / 'baseline', train, validation)
    candidates = [('baseline', baseline)]
    start = perf_counter()
    if kind != 'dummy':
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED),
                                   storage=f'sqlite:///{(folder / "study.sqlite3").as_posix()}', study_name=name)
        def objective(trial):
            run_id = f'trial_{trial.number:02d}'
            params = search_parameters(trial, kind)
            try:
                info = fit_candidate(kind, weighted, params, folder / run_id, train, validation)
            except Exception as error:
                trial.set_user_attr('failure', f'{type(error).__name__}: {error}')
                write_json(folder / run_id / 'failure.json', {'error': str(error), 'parameters': params})
                raise
            candidates.append((run_id, info))
            return info['validation_macro_f1']
        try:
            study.optimize(objective, n_trials=trials, catch=(RuntimeError, ValueError), n_jobs=1)
        finally:
            study.trials_dataframe().to_csv(folder / 'trials.csv', index=False)
    selected_id, info = max(candidates, key=lambda pair: pair[1]['validation_macro_f1'])
    selected = {'configuration': name, 'candidate': selected_id,
                'validation_macro_f1': info['validation_macro_f1'],
                'baseline_macro_f1': baseline['validation_macro_f1'],
                'tuning_seconds': perf_counter() - start, 'prepared_hash': info['prepared_hash']}
    write_json(folder / 'selected.json', selected)
    return selected
