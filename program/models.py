import joblib
import numpy as np
import torch
from torch import nn
import xgboost as xgb


class ECGNetwork(nn.Module):
    def __init__(self, kind, channels, kernel=7, hidden=64, dropout=.3):
        super().__init__()
        layers, previous = [], 1
        for width in channels:
            layers += [nn.Conv1d(previous, width, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)]
            previous = width
        self.conv = nn.Sequential(*layers)
        self.kind = kind
        if kind == 'cnn_bilstm':
            self.lstm = nn.LSTM(previous, hidden, batch_first=True, bidirectional=True)
            previous = hidden * 2
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(previous, 5))

    def forward(self, x):
        x = self.conv(x)
        if self.kind == 'cnn_bilstm':
            _, (hidden, _) = self.lstm(x.transpose(1, 2))
            x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            x = x.mean(dim=2)
        return self.head(x)


def network(kind, parameters):
    return ECGNetwork(kind, parameters['channels'], parameters.get('kernel', 7),
                      parameters.get('hidden', 64), parameters.get('dropout', .3))


def load_model(folder, info, device='cpu'):
    kind = info['kind']
    if kind in ['rf', 'dummy']:
        return joblib.load(folder / 'model.joblib')
    if kind == 'xgb':
        model = xgb.Booster()
        model.load_model(folder / 'model.ubj')
        return model
    model = network(kind, info['parameters'])
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
    return model.to(device).eval()


def probabilities(model, kind, values, device='cpu', batch_size=128):
    if kind in ['rf', 'dummy']:
        output = np.zeros((len(values), 5), dtype=np.float32)
        output[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(values)
        return output
    if kind == 'xgb':
        return model.predict(xgb.DMatrix(values))
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            # waveform: (beats, 252) -> Conv1d: (beats, 1, 252)
            batch = torch.as_tensor(values[start:start + batch_size, None, :], device=device)
            outputs.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
